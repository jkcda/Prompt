"""镜头运动分析：给每个镜头一组**客观的运动数字**。

为什么必须有这一步
------------------
单帧图像里没有运动信息。实测一段有运镜的偶像 MV，38 个镜头**全部**被标成
`static` —— 模型只能靠「同一镜头内多帧的构图变化」去猜，而它拿不准的时候
一律选择最保守的答案。把「相机到底动没动」交给模型猜，结果就是全片静止。

所以这里用 ffmpeg + 纯 Python 算出几样东西，作为**数据**喂给模型：

    shift_per_frame  帧间画面位移（分析像素）与方向 → 摇 / 俯仰
    divergence       左右半幅位移之差 → 推近 / 拉远（缩放让位移左右反号）
    px_per_sec       位移速度，与采样帧率无关，可跨镜头比较
    total_shift      整个镜头的累计位移，判断「动没动」用它最公平
    amplitude        帧间平均亮度变化，反映画面变化的剧烈程度

模型据此写运镜，并且**只有 `camera_static` 为真时才允许写 static**。

为什么不引入 OpenCV / numpy
---------------------------
本机没装，为了这一个功能装几十 MB 的依赖不值得（项目其它部分全是零额外依赖）。
ffmpeg 能把帧导成 rawvideo 灰度小图，剩下的用纯 Python 算**全局**
Lucas-Kanade（亮度梯度法）就够 —— 我们要的是一个位移量级和方向，
不是稠密光流。

⚠️ 纯梯度法假设位移小于 1 像素，实测对 4.4 px/帧 的摇镜只能解出 2.4。
所以走了三步：
    1. 在最粗的层做**整数位移暴力搜索**定初值（对周期纹理也不会混叠）
    2. 金字塔逐层细化（每层先 warp 再解残差，每层面对的位移都 < 1 像素）
    3. 帧对之间传递位移作初值（同一镜头内运镜是连续的）
实测 3 像素以内的位移能解到 0.1 像素精度，8 像素以上仍会偏，
所以 `suggest_camera` 在位移过大时只报方向、不报精确量。
"""

from __future__ import annotations

import logging
import math
import statistics
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from ..core.config import ffmpeg_required

log = logging.getLogger("motion")

# 分析用的帧尺寸。16:9 与视频常见画幅一致；竖屏会被拉伸，但方向判断不受影响
# （拉伸是各向异性的固定变换，pan/tilt 的相对大小关系保持）。
_ANALYSIS_W = 160
_ANALYSIS_H = 90
# 镜头内取样帧率。太密对判断运镜没有额外信息，只是浪费 CPU。
_SAMPLE_FPS = 6.0
# 单个镜头最多取多少帧（长镜头按这个反算取样帧率，避免 60 秒镜头抽 360 帧）。
_MAX_SAMPLES = 48
# 单次 ffmpeg 调用的超时
_TIMEOUT = 120.0

# 迭代 warp 的最大次数。位移小的时候第一次就收敛了，只有快速运镜才需要多迭代。
_MAX_ITERS = 5
# 迭代收敛判据（像素）
_CONVERGED = 0.05

# 判「相机没动」的门槛（分析像素，画面宽 160）。
#
# ⚠️ 主判据是**方向一致性**，不是位移大小。实测一段真实素材的静止镜头会测出
# 1.2~2.6 px/s 的假位移（压缩噪声 + 微抖），只看大小根本分不出来；而它们的
# 方向一致性都在 0.55~0.70（随机），真实运镜是 0.87。
#
#   一致性 ≥ _CONSISTENT_DIRECTION 且每帧位移 ≥ _MIN_VISIBLE_SHIFT → 确定是运镜
#   位移 ≥ _STRONG_MOTION_PX_PER_SEC（大到不可能是噪声）→ 也算运镜（兜底）
#   其余 → 静止
_CONSISTENT_DIRECTION = 0.75
_MIN_VISIBLE_SHIFT = 0.15      # 每帧位移（分析像素），低于此值肉眼看不出来
_STRONG_MOTION_PX_PER_SEC = 8.0
# 缩放（推拉）明显时也是运动。左右半幅位移差到 2 像素已经是可辨的推拉。
_STRONG_DIVERGENCE = 2.0


@dataclass
class ShotMotion:
    """一个镜头的客观运动测量结果。"""

    shot_index: int
    amplitude: float = 0.0        # 帧间平均亮度变化（%），反映画面变化剧烈程度
    shift_per_frame: float = 0.0  # 帧间位移（分析像素）
    px_per_sec: float = 0.0       # 位移速度（分析像素/秒），与采样帧率无关
    total_shift: float = 0.0      # 整个镜头的累计位移（分析像素）
    u: float = 0.0                # 水平位移分量，正 = 内容右移
    v: float = 0.0                # 垂直位移分量，正 = 内容下移
    divergence: float = 0.0       # 左右半幅水平位移差，正 = 内容向外扩散（推近）
    vertical_divergence: float = 0.0
    fps: float = 0.0              # 实际采样帧率
    samples: int = 0              # 实际参与比较的帧对数
    texture: float = 0.0          # 平均梯度强度，太低说明画面平坦、位移估计不可信
    # 位移方向的**一致性**（0~1）：有多少帧对的位移方向与整体中位数方向一致。
    #
    # 这是区分「慢速运镜」和「静止 + 噪声」的关键指标，比位移大小可靠得多：
    #   运镜 → 每一帧都往同一个方向走，一致性接近 1
    #   噪声 → 方向随机，一致性接近 0.5
    # 实测一段静止画面会测出 2 px/s 的假位移，只看大小根本分不出来。
    direction_consistency: float = 0.0
    # 帧间**显著变化**的像素占比（0~1），以及变化区重心位移。
    #
    # 为什么单独测它：单帧里「走路的人」和「站着的人」几乎一样 ——
    # 主体位移是静止图像唯一表达不出来的东西。但两帧一比就有了：
    # 相机不动时，画面里显著变化的那些像素就是主体在动。
    # 实测模型会把走动的人写成 "she stands"，就是因为只给了它单帧印象。
    subject_change: float = 0.0
    subject_dx: float = 0.0
    subject_dy: float = 0.0
    error: str = ""               # 抽取/分析失败的原因（不抛异常，交给调用方决定）

    @property
    def subject_moving(self) -> bool:
        """主体是否在动（**与相机无关**）。

        相机静止时，画面里显著变化的像素就是主体。阈值取 2%：
        实测走动/转身会带来 5%~20% 的显著变化，而压缩噪声在 1% 以下。
        """
        return self.subject_change >= 0.02

    # ---------- 判定 ----------

    @property
    def camera_static(self) -> bool:
        """相机是否真的没动。

        ⚠️ 判定**只看位移**，不看 `amplitude`。
        主体在跳舞而相机架在三脚架上，amplitude 很高但位移接近 0 ——
        那是「相机静止、主体在动」，运镜仍然是 static。反过来相机在推轨时
        位移显著，即使画面里没什么东西在动，也不能写 static。

        判据分两层：
          1. 方向一致（≥ `_CONSISTENT_DIRECTION`）且有可见位移 → 确定是运镜；
          2. 位移大到不可能是噪声（或缩放明显）→ 也算运镜；
          3. 其余一律判静止。
        """
        if self.samples <= 0:
            return False
        # 纹理太弱时位移估计不可信，宁可判「不确定」也不能误报 static
        if self.texture < 1.5:
            return False

        if (
            self.direction_consistency >= _CONSISTENT_DIRECTION
            and self.shift_per_frame >= _MIN_VISIBLE_SHIFT
        ):
            return False

        if self.px_per_sec >= _STRONG_MOTION_PX_PER_SEC:
            return False
        return abs(self.divergence) < _STRONG_DIVERGENCE


# ---------------------------------------------------------------------------
# 抽帧
# ---------------------------------------------------------------------------

def _extract_gray(src: str, start: float, end: float, fps: float) -> tuple[list[bytes], str]:
    """抽一段灰度 rawvideo，返回 (帧字节列表, 失败原因)。

    每帧恰好 `_ANALYSIS_W * _ANALYSIS_H` 字节（灰度 1 字节/像素）。
    """
    span = max(0.05, end - start)
    args = [
        ffmpeg_required(), "-hide_banner", "-nostats", "-loglevel", "error",
        # -ss 放在 -i 之前走快速定位。运动分析不需要帧精确 —— 起点偏几十毫秒
        # 对「推/摇/移」的判断毫无影响，但能省掉一次全片解码。
        "-ss", f"{max(0.0, start):.3f}",
        "-i", str(src),
        "-t", f"{span:.3f}",
        "-vf", (
            f"fps={fps:.3f},"
            f"scale={_ANALYSIS_W}:{_ANALYSIS_H}:flags=area,"
            "format=gray"
        ),
        "-f", "rawvideo", "-pix_fmt", "gray", "-",
    ]
    try:
        cp = subprocess.run(args, capture_output=True, timeout=_TIMEOUT)
    except subprocess.TimeoutExpired:
        return [], f"运动分析抽帧超时（{span:.1f}s 片段）"
    except Exception as exc:  # noqa: BLE001
        return [], f"运动分析抽帧异常：{exc}"

    raw = cp.stdout or b""
    size = _ANALYSIS_W * _ANALYSIS_H
    n = len(raw) // size
    if n < 2:
        return [], f"可用帧不足（{n} 帧）"
    return [raw[i * size:(i + 1) * size] for i in range(n)], ""


# ---------------------------------------------------------------------------
# 位移估计
# ---------------------------------------------------------------------------

def _estimate_shift_raw(
    a: bytes, b: bytes, w: int, h: int, x0: int, x1: int, y0: int, y1: int
) -> tuple[float, float, float, float]:
    """**单尺度**的全局位移估计，返回 (u, v, mae, texture)。

    解亮度恒常方程 Ix·u + Iy·v + It = 0 的最小二乘解：

        [[ΣIx², ΣIxIy], [ΣIxIy, ΣIy²]] · [u, v]ᵀ = -[ΣIxIt, ΣIyIt]ᵀ

    约定：返回的 (u, v) 是 **b 的内容相对于 a 的位移**（u>0 = 内容右移）。

    ⚠️ 线性化假设位移小于一个像素，位移大时会严重低估，
    所以对外一律走 `_pyramid_shift`。
    """
    sxx = sxy = syy = sxt = syt = 0.0
    mae_sum = 0.0
    grad_sum = 0.0
    n = 0

    for y in range(max(1, y0), min(h - 1, y1)):
        base = y * w
        for x in range(max(1, x0), min(w - 1, x1)):
            i = base + x
            # 用前后两帧梯度的平均，比只用一帧稳
            ix = (a[i + 1] - a[i - 1] + b[i + 1] - b[i - 1]) * 0.25
            iy = (a[i + w] - a[i - w] + b[i + w] - b[i - w]) * 0.25
            it = b[i] - a[i]
            sxx += ix * ix
            sxy += ix * iy
            syy += iy * iy
            sxt += ix * it
            syt += iy * it
            mae_sum += abs(it)
            grad_sum += abs(ix) + abs(iy)
            n += 1

    if n == 0:
        return 0.0, 0.0, 0.0, 0.0

    det = sxx * syy - sxy * sxy
    mae = mae_sum / n
    texture = grad_sum / n
    # det 太小 = 区域里没有可用纹理，解出来的 (u, v) 是噪声
    if det < 1e-3 or texture < 0.5:
        return 0.0, 0.0, mae, texture

    u = (-syy * sxt + sxy * syt) / det
    v = (sxy * sxt - sxx * syt) / det
    return u, v, mae, texture


def _warp(src: bytes, w: int, h: int, dx: float, dy: float) -> bytes:
    """把图像按 (dx, dy) 平移，越界部分用**边缘像素复制**。

    只取位移的整数部分 —— 小数部分留给下一次梯度法迭代去补，
    配合起来精度足够，而且比双线性插值快得多。

    ⚠️ 越界必须 clamp 而不是填 0。填 0 会在边缘造出一整条黑边，
    而黑边与真实内容的差异极大，会把下一步的最小二乘解带偏 ——
    垂直位移时丢的是整行（160 个像素），污染比水平位移严重得多。
    """
    ix, iy = int(round(dx)), int(round(dy))
    if ix == 0 and iy == 0:
        return src
    out = bytearray(w * h)
    for y in range(h):
        sy = min(h - 1, max(0, y + iy))
        s_base = sy * w
        d_base = y * w
        for x in range(w):
            sx = min(w - 1, max(0, x + ix))
            out[d_base + x] = src[s_base + sx]
    return bytes(out)


def _solve_shift(
    a: bytes, b: bytes, w: int, h: int, x0: int, x1: int, y0: int, y1: int
) -> tuple[float, float, float, float]:
    """迭代 warp 的位移估计，返回 (u, v, mae, texture)。

    单次梯度法只能处理亚像素位移（实测 8 像素的真实位移只解出 4）。做法是
    「解 → 按解出的位移把 b warp 过去 → 再解残差 → 累加」，迭代到收敛。
    每次迭代面对的残差都比上一次小，几轮下来就能逼近真实值。

    ⚠️ 试过用图像金字塔（多尺度）代替迭代：金字塔在**周期性纹理**上会混叠到
    错误的峰 —— 实测一张正弦条纹图里 8 像素的位移被解成 28.9。
    迭代 warp 是连续逼近，不会跳到别的周期上，对真实素材更稳。
    """
    u = v = 0.0
    mae = tex = 0.0
    for _ in range(_MAX_ITERS):
        warped = _warp(b, w, h, u, v)
        du, dv, mae, tex = _estimate_shift_raw(a, warped, w, h, x0, x1, y0, y1)
        u += du
        v += dv
        if abs(du) < _CONVERGED and abs(dv) < _CONVERGED:
            break
    return u, v, mae, tex


@dataclass
class _PairResult:
    u: float
    v: float
    divergence: float
    vertical_divergence: float
    mae: float
    texture: float
    # 帧间**显著变化**的像素占比（0~1）。
    #
    # 为什么需要它：单帧里「走路的人」和「站着的人」几乎一样 —— 主体位移是
    # 静止图像唯一表达不出来的东西。但两帧一比就有了：相机不动时，画面里
    # 变化的那些像素就是主体在动。实测模型会把走动的人写成 "she stands"，
    # 就是因为只给了它单帧印象。
    #
    # 用阈值化差异（只算明显变化的像素）而不是整体 MAE，否则光照渐变和
    # 压缩噪声会把它抬高。
    change_ratio: float = 0.0
    # 变化区域的加权重心**位置**（归一化到画面宽高，0.5 = 画面中心）。
    #
    # ⚠️ 这是位置不是位移。位移要拿相邻帧对的这个值相减才算得出来
    # （见 `analyze_shot_motion`）。踩过：一开始把它当位移直接报出去，
    # 结果每个镜头都显示「向右移动 0.48」—— 其实那只是「变化发生在画面中间」。
    change_cx: float = 0.5
    change_cy: float = 0.5


def _change_stats(a: bytes, b: bytes, w: int, h: int, threshold: int = 18) -> tuple[float, float, float]:
    """帧间显著变化：占比 + 变化区重心位置。

    扣掉相机位移这一步由调用方负责（相机静止时直接用原始帧对即可）。
    """
    total = 0
    sx = sy = 0.0
    weight = 0.0
    n = w * h
    for i in range(n):
        d = a[i] - b[i]
        if d < 0:
            d = -d
        if d > threshold:
            total += 1
            x = i % w
            y = i // w
            sx += x * d
            sy += y * d
            weight += d
    if weight <= 0 or n == 0:
        return total / n, 0.5, 0.5
    # 重心：变化剧烈的像素权重更大
    return total / n, (sx / weight) / w, (sy / weight) / h


def _measure_pair(a: bytes, b: bytes, w: int, h: int) -> _PairResult:
    """测一对帧的位移。

    ⚠️ 用**四个角**分别估计再取中位数，而不是整幅一起估。
    全局估计有个致命弱点：主体在画面里移动会被当成相机运动。
    实测一段主体在动的静止镜头（相机固定在三脚架上），整幅估计报出
    「9.5 像素的位移」，于是被判成有运镜。分四角之后，主体只污染其中
    一两个角，中位数不受影响。

    四角的面积各占 1/4，合计计算量与整幅相同，没有额外开销。
    顺带还能直接算出推拉所需的左右/上下半幅位移差。
    """
    mid_x, mid_y = w // 2, h // 2
    quads = (
        (0, mid_x, 0, mid_y),          # 左上
        (mid_x, w, 0, mid_y),          # 右上
        (0, mid_x, mid_y, h),          # 左下
        (mid_x, w, mid_y, h),          # 右下
    )
    res = [_solve_shift(a, b, w, h, *q) for q in quads]
    us = [r[0] for r in res]
    vs = [r[1] for r in res]

    left = (us[0] + us[2]) / 2.0
    right = (us[1] + us[3]) / 2.0
    top = (vs[0] + vs[1]) / 2.0
    bottom = (vs[2] + vs[3]) / 2.0

    med_u, med_v = _median(us), _median(vs)

    # 帧间显著变化。相机在动时先按全局位移对齐，否则差异图里混着相机运动，
    # 会把「相机移动」误报成「主体在动」。
    if abs(med_u) > 0.5 or abs(med_v) > 0.5:
        aligned = _warp(b, w, h, med_u, med_v)
    else:
        aligned = b
    change_ratio, change_cx, change_cy = _change_stats(a, aligned, w, h)

    return _PairResult(
        u=med_u,
        v=med_v,
        divergence=right - left,
        vertical_divergence=bottom - top,
        mae=_median([r[2] for r in res]),
        texture=_median([r[3] for r in res]),
        change_ratio=change_ratio,
        change_cx=change_cx,
        change_cy=change_cy,
    )


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


# ---------------------------------------------------------------------------
# 对外入口
# ---------------------------------------------------------------------------

def analyze_shot_motion(src: str, start: float, end: float, shot_index: int = 0) -> ShotMotion:
    """分析一个镜头的运动。失败时返回带 `error` 的空结果，不抛异常。"""
    result = ShotMotion(shot_index=shot_index)
    span = max(0.05, end - start)
    fps = min(_SAMPLE_FPS, max(1.0, _MAX_SAMPLES / span))

    frames, why = _extract_gray(src, start, end, fps)
    if not frames:
        result.error = why
        return result

    W, H = _ANALYSIS_W, _ANALYSIS_H

    us: list[float] = []
    vs: list[float] = []
    divs: list[float] = []
    vdivs: list[float] = []
    maes: list[float] = []
    textures: list[float] = []
    changes: list[float] = []
    cxs: list[float] = []
    cys: list[float] = []

    for i in range(len(frames) - 1):
        p = _measure_pair(frames[i], frames[i + 1], W, H)
        us.append(p.u)
        vs.append(p.v)
        divs.append(p.divergence)
        vdivs.append(p.vertical_divergence)
        maes.append(p.mae)
        textures.append(p.texture)
        changes.append(p.change_ratio)
        cxs.append(p.change_cx)
        cys.append(p.change_cy)

    result.samples = len(maes)
    result.fps = round(fps, 2)
    result.u = round(_median(us), 3)
    result.v = round(_median(vs), 3)
    result.shift_per_frame = round(math.hypot(result.u, result.v), 3)
    result.px_per_sec = round(result.shift_per_frame * fps, 2)
    result.total_shift = round(result.px_per_sec * span, 2)
    result.divergence = round(_median(divs), 3)
    result.vertical_divergence = round(_median(vdivs), 3)
    result.amplitude = round(_median(maes) / 255.0 * 100.0, 2)
    result.texture = round(_median(textures), 2)
    result.direction_consistency = round(_direction_consistency(us, vs), 3)
    result.subject_change = round(_median(changes), 4)
    # 变化区重心的**相邻帧差**才是位移（重心本身只是位置）
    dxs = [cxs[i + 1] - cxs[i] for i in range(len(cxs) - 1)]
    dys = [cys[i + 1] - cys[i] for i in range(len(cys) - 1)]
    result.subject_dx = round(_median(dxs), 4)
    result.subject_dy = round(_median(dys), 4)
    return result


def _direction_consistency(us: list[float], vs: list[float]) -> float:
    """有多少帧对的位移方向与整体中位数方向一致（点积 > 0）。

    这是区分「慢速运镜」和「静止 + 噪声」的关键：运镜的每一帧都往同一个
    方向走，一致性接近 1；噪声方向随机，一致性在 0.5 附近。
    只看位移大小是分不出来的 —— 实测静止画面也会测出 2 px/s 的假位移。
    """
    if len(us) < 3:
        return 0.0
    mu, mv = _median(us), _median(vs)
    if math.hypot(mu, mv) < 1e-6:
        return 0.0
    agree = sum(1 for u, v in zip(us, vs, strict=True) if u * mu + v * mv > 0)
    return agree / len(us)


def analyze_shots_motion(
    src: str,
    shots: list[tuple[float, float]],
    workers: int = 4,
) -> list[ShotMotion]:
    """批量分析所有镜头（并行）。任何单个镜头失败都不影响其它镜头。"""
    if not shots:
        return []

    out: list[ShotMotion | None] = [None] * len(shots)

    def _one(idx: int) -> None:
        a, b = shots[idx]
        try:
            out[idx] = analyze_shot_motion(src, a, b, shot_index=idx)
        except Exception as exc:  # noqa: BLE001
            # 运动分析是**增强**，不该拖垮管线。失败就退回「模型自己判断」。
            log.warning("镜头 %d 运动分析失败: %s", idx + 1, exc)
            m = ShotMotion(shot_index=idx)
            m.error = str(exc)
            out[idx] = m

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        list(pool.map(_one, range(len(shots))))

    return [m or ShotMotion(shot_index=i) for i, m in enumerate(out)]


def suggest_camera(m: ShotMotion) -> str:
    """把测量结果翻成一个运镜类型建议。仅作参考，模型仍可给出更细的描述。"""
    if m.error or m.samples <= 0:
        return "unknown"
    if m.camera_static:
        return "static"
    if m.texture < 1.5:
        return "static? (low texture, shift unreliable)"

    # 把水平位移拆成「平移」和「缩放」两个分量：
    #   divergence = u_right - u_left
    #   缩放（推/拉）让左右半幅的位移**反号**，平移让它们**同号**。
    #   所以平移分量就是全幅位移 u，缩放分量是半幅差的一半。
    # ⚠️ 不能直接拿 divergence 和 u 比大小 —— divergence 是两倍的关系，
    # 而且缩放滤镜的整数取整会额外制造一点假平移。用 0.5 倍做门槛。
    zoom = m.divergence / 2.0
    vzoom = m.vertical_divergence / 2.0
    horiz = abs(m.u)
    vert = abs(m.v)

    if abs(zoom) >= 0.35 and abs(zoom) >= 0.5 * horiz:
        return "dolly-in" if zoom > 0 else "dolly-out"
    # ⚠️ 垂直方向的门槛要比水平高得多。上下半幅的垂直位移估计本身就带噪声
    # （实测纯摇镜的 vertical_divergence 能到 ±0.7），用同样的门槛会把
    # 摇镜误判成升降。而且 2D 位移本来就分不出「相机升降」和「变焦」，
    # 只在信号很强时才报。
    if abs(vzoom) >= 0.8 and abs(vzoom) >= abs(zoom):
        return "crane-up" if vzoom > 0 else "crane-down"

    if horiz >= vert:
        return "pan-left" if m.u > 0 else "pan-right"   # 内容右移 = 相机向左摇
    return "tilt-up" if m.v > 0 else "tilt-down"        # 内容下移 = 相机向上摇


def describe_motion(m: ShotMotion, index: int | None = None) -> str:
    """给模型看的一行英文说明。数字 + 判定建议，两者都给。

    为什么不只给判定：模型看到「pan-left」可能照抄成固定标签，而给它
    「画面内容右移 1.8 像素/帧」它才能自己判断幅度是缓慢还是快速。
    """
    label = f"shot {index + 1}" if index is not None else "shot"
    if m.error:
        return f"{label}: motion measurement unavailable ({m.error})"
    if m.samples <= 0:
        return f"{label}: motion measurement unavailable (too few frames)"

    kind = suggest_camera(m)
    parts = [
        f"{label}: {kind}",
        f"content moves {m.u:+.2f} px horizontally and {m.v:+.2f} px vertically per frame "
        f"(positive = right / down), i.e. {m.px_per_sec:.1f} px per second",
        f"total movement across the whole shot: {m.total_shift:.1f} px "
        f"({'effectively no camera movement' if m.camera_static else 'the camera really moves'})",
        f"direction consistency {m.direction_consistency:.2f} "
        f"(high = one continuous camera move; near 0.5 = random jitter, not a move)",
        f"horizontal scale change {m.divergence:+.2f} "
        f"(positive = content spreading outward = camera moving closer)",
        f"frame-to-frame brightness change {m.amplitude:.2f}%",
    ]
    # 主体是否在动 —— 这是单帧**看不出来**、只有对比帧才知道的信息。
    # 必须显式说出来：实测模型会默认写 "she stands"，因为它从静态帧里
    # 判断不出人在走路（走路的人和站着的人在单帧里几乎一样）。
    if m.subject_moving:
        parts.append(
            f"SUBJECT MOVEMENT: {m.subject_change * 100:.1f}% of the frame changes between "
            "consecutive frames even after compensating for camera motion — the subject is "
            "genuinely moving (walking, turning, gesturing, shifting weight). "
            "Do NOT describe it as standing still. "
            f"(The change centre drifts {m.subject_dx:+.4f} right / {m.subject_dy:+.4f} down "
            "per frame — that is a weak hint only; read the real direction from the frames.)"
        )
    else:
        parts.append(
            f"SUBJECT MOVEMENT: only {m.subject_change * 100:.1f}% of the frame changes between "
            "frames — beyond the camera, very little is moving."
        )
    if m.texture < 1.5:
        parts.append(
            f"WARNING: low image texture ({m.texture:.1f}) — the movement estimate is "
            "unreliable here, judge the camera from the frames instead"
        )
    return "; ".join(parts)


def summarize_for_prompt(motions: list[ShotMotion]) -> str:
    """拼成给模型的整段说明。"""
    return "\n".join(describe_motion(m, i) for i, m in enumerate(motions))
