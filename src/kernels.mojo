"""Compute kernels for the covered two-dimensional scikit-image API."""

from std.algorithm import sync_parallelize
from std.math import abs, floor, sqrt
from std.sys import simd_width_of

comptime FPtr = UnsafePointer[Float64, AnyOrigin[mut=True]]
comptime IPtr = UnsafePointer[Int64, AnyOrigin[mut=True]]
comptime BPtr = UnsafePointer[UInt8, AnyOrigin[mut=True]]
comptime PARALLEL_THRESHOLD = 262144
comptime MAX_ROW_TASKS = 32


def fp(addr: Int) -> FPtr:
    return FPtr(unsafe_from_address=addr)


def ip(addr: Int) -> IPtr:
    return IPtr(unsafe_from_address=addr)


def bp(addr: Int) -> BPtr:
    return BPtr(unsafe_from_address=addr)


def border_index(i: Int, n: Int, mode: Int) -> Int:
    if i >= 0 and i < n:
        return i
    if mode == 1:
        return -1
    if n == 1:
        return 0
    if mode == 2:
        return 0 if i < 0 else n - 1
    if mode == 4:
        var j = i % n
        return j + n if j < 0 else j
    var j = i
    if mode == 0:
        while j < 0 or j >= n:
            j = -j - 1 if j < 0 else 2 * n - j - 1
    else:
        while j < 0 or j >= n:
            j = -j if j < 0 else 2 * n - j - 2
    return j


def sample(src: FPtr, y: Int, x: Int, h: Int, w: Int, mode: Int, cval: Float64) -> Float64:
    var yy = border_index(y, h, mode)
    var xx = border_index(x, w, mode)
    if yy < 0 or xx < 0:
        return cval
    return src[yy * w + xx]


def sample_u8(
    src: BPtr, y: Int, x: Int, h: Int, w: Int, mode: Int, cval: UInt8
) -> UInt8:
    var yy = border_index(y, h, mode)
    var xx = border_index(x, w, mode)
    if yy < 0 or xx < 0:
        return cval
    return src[yy * w + xx]


@export("msi_convolve_axis")
def msi_convolve_axis(
    src_addr: Int,
    dst_addr: Int,
    kernel_addr: Int,
    h: Int,
    w: Int,
    size: Int,
    axis: Int,
    mode: Int,
    cval: Float64,
) abi("C"):
    var src = fp(src_addr)
    var dst = fp(dst_addr)
    var kernel = fp(kernel_addr)
    var radius = size // 2
    comptime W = simd_width_of[DType.float64]()

    @parameter
    def row(y: Int):
        var x_start = min(radius, w) if axis == 1 else 0
        var x_end = (
            max(x_start, w - (size - radius - 1))
            if axis == 1
            else w
        )
        for x in range(x_start):
            var acc = 0.0
            for k in range(size):
                var offset = k - radius
                var yy = y + offset if axis == 0 else y
                var xx = x + offset if axis == 1 else x
                acc += kernel[k] * sample(src, yy, xx, h, w, mode, cval)
            dst[y * w + x] = acc
        var x = x_start
        while x + W <= x_end:
            var acc = SIMD[DType.float64, W](0.0)
            for k in range(size):
                if axis == 0:
                    var yy = border_index(y + k - radius, h, mode)
                    if yy < 0:
                        acc += kernel[k] * cval
                    else:
                        acc += kernel[k] * src.load[width=W](yy * w + x)
                else:
                    acc += kernel[k] * src.load[width=W](
                        y * w + x + k - radius
                    )
            dst.store(y * w + x, acc)
            x += W
        while x < w:
            var acc = 0.0
            for k in range(size):
                var offset = k - radius
                var yy = y + offset if axis == 0 else y
                var xx = x + offset if axis == 1 else x
                acc += kernel[k] * sample(src, yy, xx, h, w, mode, cval)
            dst[y * w + x] = acc
            x += 1

    if h * w < PARALLEL_THRESHOLD or h == 1:
        for y in range(h):
            row(y)
    else:
        var tasks = min(h, MAX_ROW_TASKS)

        @parameter
        def rows(task: Int):
            var start = task * h // tasks
            var end = (task + 1) * h // tasks
            for y in range(start, end):
                row(y)

        sync_parallelize[rows](tasks)


def sobel_value(
    src: FPtr,
    y: Int,
    x: Int,
    h: Int,
    w: Int,
    axis: Int,
    mode: Int,
    cval: Float64,
) -> Float64:
    if axis == 0:
        return (
            -sample(src, y - 1, x - 1, h, w, mode, cval)
            - 2.0 * sample(src, y - 1, x, h, w, mode, cval)
            - sample(src, y - 1, x + 1, h, w, mode, cval)
            + sample(src, y + 1, x - 1, h, w, mode, cval)
            + 2.0 * sample(src, y + 1, x, h, w, mode, cval)
            + sample(src, y + 1, x + 1, h, w, mode, cval)
        ) * 0.25
    return (
        -sample(src, y - 1, x - 1, h, w, mode, cval)
        - 2.0 * sample(src, y, x - 1, h, w, mode, cval)
        - sample(src, y + 1, x - 1, h, w, mode, cval)
        + sample(src, y - 1, x + 1, h, w, mode, cval)
        + 2.0 * sample(src, y, x + 1, h, w, mode, cval)
        + sample(src, y + 1, x + 1, h, w, mode, cval)
    ) * 0.25


@export("msi_sobel")
def msi_sobel(
    src_addr: Int,
    dst_addr: Int,
    h: Int,
    w: Int,
    axis: Int,
    mode: Int,
    cval: Float64,
) abi("C"):
    var src = fp(src_addr)
    var dst = fp(dst_addr)
    comptime W = simd_width_of[DType.float64]()

    @parameter
    def row(y: Int):
        if w == 1:
            dst[y * w] = sobel_value(src, y, 0, h, w, axis, mode, cval)
            return
        dst[y * w] = sobel_value(src, y, 0, h, w, axis, mode, cval)
        var ym = border_index(y - 1, h, mode)
        var yp = border_index(y + 1, h, mode)
        var x = 1
        while x + W <= w - 1:
            var top_left = SIMD[DType.float64, W](cval)
            var top_mid = SIMD[DType.float64, W](cval)
            var top_right = SIMD[DType.float64, W](cval)
            var bottom_left = SIMD[DType.float64, W](cval)
            var bottom_mid = SIMD[DType.float64, W](cval)
            var bottom_right = SIMD[DType.float64, W](cval)
            if ym >= 0:
                top_left = src.load[width=W](ym * w + x - 1)
                top_mid = src.load[width=W](ym * w + x)
                top_right = src.load[width=W](ym * w + x + 1)
            if yp >= 0:
                bottom_left = src.load[width=W](yp * w + x - 1)
                bottom_mid = src.load[width=W](yp * w + x)
                bottom_right = src.load[width=W](yp * w + x + 1)
            var edge = SIMD[DType.float64, W](0.0)
            if axis == 0:
                edge = (
                    -top_left - 2.0 * top_mid - top_right
                    + bottom_left + 2.0 * bottom_mid + bottom_right
                ) * 0.25
            else:
                var middle_left = src.load[width=W](y * w + x - 1)
                var middle_right = src.load[width=W](y * w + x + 1)
                edge = (
                    -top_left - 2.0 * middle_left - bottom_left
                    + top_right + 2.0 * middle_right + bottom_right
                ) * 0.25
            dst.store(y * w + x, edge)
            x += W
        while x < w - 1:
            dst[y * w + x] = sobel_value(src, y, x, h, w, axis, mode, cval)
            x += 1
        dst[y * w + w - 1] = sobel_value(
            src, y, w - 1, h, w, axis, mode, cval
        )

    if h * w < PARALLEL_THRESHOLD or h == 1:
        for y in range(h):
            row(y)
    else:
        var tasks = min(h, MAX_ROW_TASKS)

        @parameter
        def rows(task: Int):
            var start = task * h // tasks
            var end = (task + 1) * h // tasks
            for y in range(start, end):
                row(y)

        sync_parallelize[rows](tasks)


@export("msi_sobel_magnitude")
def msi_sobel_magnitude(
    src_addr: Int,
    dst_addr: Int,
    h: Int,
    w: Int,
    mode: Int,
    cval: Float64,
) abi("C"):
    var src = fp(src_addr)
    var dst = fp(dst_addr)
    comptime W = simd_width_of[DType.float64]()

    @parameter
    def row(y: Int):
        if w == 1:
            var gy = sobel_value(src, y, 0, h, w, 0, mode, cval)
            var gx = sobel_value(src, y, 0, h, w, 1, mode, cval)
            dst[y * w] = sqrt((gy * gy + gx * gx) * 0.5)
            return
        var gy0 = sobel_value(src, y, 0, h, w, 0, mode, cval)
        var gx0 = sobel_value(src, y, 0, h, w, 1, mode, cval)
        dst[y * w] = sqrt((gy0 * gy0 + gx0 * gx0) * 0.5)
        var ym = border_index(y - 1, h, mode)
        var yp = border_index(y + 1, h, mode)
        var x = 1
        while x + W <= w - 1:
            var top_left = SIMD[DType.float64, W](cval)
            var top_mid = SIMD[DType.float64, W](cval)
            var top_right = SIMD[DType.float64, W](cval)
            var bottom_left = SIMD[DType.float64, W](cval)
            var bottom_mid = SIMD[DType.float64, W](cval)
            var bottom_right = SIMD[DType.float64, W](cval)
            if ym >= 0:
                top_left = src.load[width=W](ym * w + x - 1)
                top_mid = src.load[width=W](ym * w + x)
                top_right = src.load[width=W](ym * w + x + 1)
            if yp >= 0:
                bottom_left = src.load[width=W](yp * w + x - 1)
                bottom_mid = src.load[width=W](yp * w + x)
                bottom_right = src.load[width=W](yp * w + x + 1)
            var middle_left = src.load[width=W](y * w + x - 1)
            var middle_right = src.load[width=W](y * w + x + 1)
            var gy = (
                -top_left - 2.0 * top_mid - top_right
                + bottom_left + 2.0 * bottom_mid + bottom_right
            ) * 0.25
            var gx = (
                -top_left - 2.0 * middle_left - bottom_left
                + top_right + 2.0 * middle_right + bottom_right
            ) * 0.25
            dst.store(y * w + x, sqrt((gy * gy + gx * gx) * 0.5))
            x += W
        while x < w - 1:
            var gy = sobel_value(src, y, x, h, w, 0, mode, cval)
            var gx = sobel_value(src, y, x, h, w, 1, mode, cval)
            dst[y * w + x] = sqrt((gy * gy + gx * gx) * 0.5)
            x += 1
        var gy_last = sobel_value(src, y, w - 1, h, w, 0, mode, cval)
        var gx_last = sobel_value(src, y, w - 1, h, w, 1, mode, cval)
        dst[y * w + w - 1] = sqrt(
            (gy_last * gy_last + gx_last * gx_last) * 0.5
        )

    if h * w < PARALLEL_THRESHOLD or h == 1:
        for y in range(h):
            row(y)
    else:
        var tasks = min(h, MAX_ROW_TASKS)

        @parameter
        def rows(task: Int):
            var start = task * h // tasks
            var end = (task + 1) * h // tasks
            for y in range(start, end):
                row(y)

        sync_parallelize[rows](tasks)


@export("msi_median")
def msi_median(
    src_addr: Int,
    dst_addr: Int,
    footprint_addr: Int,
    scratch_addr: Int,
    h: Int,
    w: Int,
    fh: Int,
    fw: Int,
    count: Int,
    mode: Int,
    cval: Float64,
) abi("C"):
    var src = fp(src_addr)
    var dst = fp(dst_addr)
    var footprint = bp(footprint_addr)
    var scratch = fp(scratch_addr)
    var ry = fh // 2
    var rx = fw // 2
    for y in range(h):
        for x in range(w):
            var n = 0
            for ky in range(fh):
                for kx in range(fw):
                    if footprint[ky * fw + kx] == 0:
                        continue
                    var value = sample(src, y + ky - ry, x + kx - rx, h, w, mode, cval)
                    var j = n
                    while j > 0 and scratch[j - 1] > value:
                        scratch[j] = scratch[j - 1]
                        j -= 1
                    scratch[j] = value
                    n += 1
            dst[y * w + x] = scratch[count // 2]


@export("msi_morph")
def msi_morph(
    src_addr: Int,
    dst_addr: Int,
    footprint_addr: Int,
    h: Int,
    w: Int,
    fh: Int,
    fw: Int,
    operation: Int,
    mode: Int,
    cval: Float64,
) abi("C"):
    var src = fp(src_addr)
    var dst = fp(dst_addr)
    var footprint = bp(footprint_addr)
    var ry = (fh - 1) // 2
    var rx = (fw - 1) // 2
    comptime W = simd_width_of[DType.float64]()

    @parameter
    def row(y: Int):
        var x_start = min(rx, w)
        var x_end = max(x_start, w - (fw - rx - 1))
        for x in range(x_start):
            var best = 1.7976931348623157e308 if operation == 0 else -1.7976931348623157e308
            for ky in range(fh):
                for kx in range(fw):
                    if footprint[ky * fw + kx] == 0:
                        continue
                    var value = sample(src, y + ky - ry, x + kx - rx, h, w, mode, cval)
                    if operation == 0:
                        if value < best:
                            best = value
                    else:
                        if value > best:
                            best = value
            dst[y * w + x] = best
        var x = x_start
        while x + W <= x_end:
            var best = SIMD[DType.float64, W](
                1.7976931348623157e308
                if operation == 0
                else -1.7976931348623157e308
            )
            for ky in range(fh):
                var yy = border_index(y + ky - ry, h, mode)
                for kx in range(fw):
                    if footprint[ky * fw + kx] == 0:
                        continue
                    var value = SIMD[DType.float64, W](cval)
                    if yy >= 0:
                        value = src.load[width=W](
                            yy * w + x + kx - rx
                        )
                    if operation == 0:
                        best = min(best, value)
                    else:
                        best = max(best, value)
            dst.store(y * w + x, best)
            x += W
        while x < w:
            var best = 1.7976931348623157e308 if operation == 0 else -1.7976931348623157e308
            for ky in range(fh):
                for kx in range(fw):
                    if footprint[ky * fw + kx] == 0:
                        continue
                    var value = sample(
                        src, y + ky - ry, x + kx - rx, h, w, mode, cval
                    )
                    if operation == 0:
                        if value < best:
                            best = value
                    else:
                        if value > best:
                            best = value
            dst[y * w + x] = best
            x += 1

    if h * w < PARALLEL_THRESHOLD or h == 1:
        for y in range(h):
            row(y)
    else:
        var tasks = min(h, MAX_ROW_TASKS)

        @parameter
        def rows(task: Int):
            var start = task * h // tasks
            var end = (task + 1) * h // tasks
            for y in range(start, end):
                row(y)

        sync_parallelize[rows](tasks)


@export("msi_morph_u8")
def msi_morph_u8(
    src_addr: Int,
    dst_addr: Int,
    footprint_addr: Int,
    h: Int,
    w: Int,
    fh: Int,
    fw: Int,
    operation: Int,
    mode: Int,
    cval_int: Int,
) abi("C"):
    var src = bp(src_addr)
    var dst = bp(dst_addr)
    var footprint = bp(footprint_addr)
    var cval = UInt8(cval_int)
    var ry = (fh - 1) // 2
    var rx = (fw - 1) // 2
    comptime W = simd_width_of[DType.uint8]()

    @parameter
    def row(y: Int):
        var x_start = min(rx, w)
        var x_end = max(x_start, w - (fw - rx - 1))
        for x in range(x_start):
            var best = UInt8(255) if operation == 0 else UInt8(0)
            for ky in range(fh):
                for kx in range(fw):
                    if footprint[ky * fw + kx] == 0:
                        continue
                    var value = sample_u8(
                        src, y + ky - ry, x + kx - rx, h, w, mode, cval
                    )
                    if operation == 0:
                        if value < best:
                            best = value
                    else:
                        if value > best:
                            best = value
            dst[y * w + x] = best
        var x = x_start
        while x + W <= x_end:
            var best = SIMD[DType.uint8, W](
                UInt8(255) if operation == 0 else UInt8(0)
            )
            for ky in range(fh):
                var yy = border_index(y + ky - ry, h, mode)
                for kx in range(fw):
                    if footprint[ky * fw + kx] == 0:
                        continue
                    var value = SIMD[DType.uint8, W](cval)
                    if yy >= 0:
                        value = src.load[width=W](
                            yy * w + x + kx - rx
                        )
                    if operation == 0:
                        best = min(best, value)
                    else:
                        best = max(best, value)
            dst.store(y * w + x, best)
            x += W
        while x < w:
            var best = UInt8(255) if operation == 0 else UInt8(0)
            for ky in range(fh):
                for kx in range(fw):
                    if footprint[ky * fw + kx] == 0:
                        continue
                    var value = sample_u8(
                        src, y + ky - ry, x + kx - rx, h, w, mode, cval
                    )
                    if operation == 0:
                        if value < best:
                            best = value
                    else:
                        if value > best:
                            best = value
            dst[y * w + x] = best
            x += 1

    if h * w < PARALLEL_THRESHOLD or h == 1:
        for y in range(h):
            row(y)
    else:
        var tasks = min(h, MAX_ROW_TASKS)

        @parameter
        def rows(task: Int):
            var start = task * h // tasks
            var end = (task + 1) * h // tasks
            for y in range(start, end):
                row(y)

        sync_parallelize[rows](tasks)


def interpolate(
    src: FPtr,
    sy: Float64,
    sx: Float64,
    h: Int,
    w: Int,
    order: Int,
    mode: Int,
    cval: Float64,
) -> Float64:
    if order == 0:
        return sample(src, Int(floor(sy + 0.5)), Int(floor(sx + 0.5)), h, w, mode, cval)
    var y0 = Int(floor(sy))
    var x0 = Int(floor(sx))
    var fy = sy - Float64(y0)
    var fx = sx - Float64(x0)
    var a = sample(src, y0, x0, h, w, mode, cval)
    var b = sample(src, y0, x0 + 1, h, w, mode, cval)
    var c = sample(src, y0 + 1, x0, h, w, mode, cval)
    var d = sample(src, y0 + 1, x0 + 1, h, w, mode, cval)
    return (a * (1.0 - fx) + b * fx) * (1.0 - fy) + (
        c * (1.0 - fx) + d * fx
    ) * fy


@export("msi_resize")
def msi_resize(
    src_addr: Int,
    dst_addr: Int,
    sh: Int,
    sw: Int,
    dh: Int,
    dw: Int,
    order: Int,
    mode: Int,
    cval: Float64,
) abi("C"):
    var src = fp(src_addr)
    var dst = fp(dst_addr)
    var scale_y = Float64(sh) / Float64(dh)
    var scale_x = Float64(sw) / Float64(dw)
    for y in range(dh):
        for x in range(dw):
            var sy = (Float64(y) + 0.5) * scale_y - 0.5
            var sx = (Float64(x) + 0.5) * scale_x - 0.5
            dst[y * dw + x] = interpolate(src, sy, sx, sh, sw, order, mode, cval)


@export("msi_warp_affine")
def msi_warp_affine(
    src_addr: Int,
    dst_addr: Int,
    matrix_addr: Int,
    sh: Int,
    sw: Int,
    dh: Int,
    dw: Int,
    order: Int,
    mode: Int,
    cval: Float64,
) abi("C"):
    var src = fp(src_addr)
    var dst = fp(dst_addr)
    var matrix = fp(matrix_addr)
    for y in range(dh):
        for x in range(dw):
            var sx = matrix[0] * Float64(x) + matrix[1] * Float64(y) + matrix[2]
            var sy = matrix[3] * Float64(x) + matrix[4] * Float64(y) + matrix[5]
            dst[y * dw + x] = interpolate(src, sy, sx, sh, sw, order, mode, cval)


@export("msi_otsu")
def msi_otsu(hist_addr: Int, centers_addr: Int, n: Int) abi("C") -> Float64:
    var hist = fp(hist_addr)
    var centers = fp(centers_addr)
    var total = 0.0
    var total_mean = 0.0
    for i in range(n):
        total += hist[i]
        total_mean += hist[i] * centers[i]
    var left_weight = 0.0
    var left_sum = 0.0
    var best_variance = -1.0
    var threshold = centers[0]
    for i in range(n - 1):
        left_weight += hist[i]
        left_sum += hist[i] * centers[i]
        var right_weight = total - left_weight
        if left_weight == 0.0 or right_weight == 0.0:
            continue
        var left_mean = left_sum / left_weight
        var right_mean = (total_mean - left_sum) / right_weight
        var delta = left_mean - right_mean
        var variance = left_weight * right_weight * delta * delta
        if variance > best_variance:
            best_variance = variance
            threshold = centers[i]
    return threshold


@export("msi_flood")
def msi_flood(
    image_addr: Int,
    result_addr: Int,
    stack_addr: Int,
    footprint_addr: Int,
    h: Int,
    w: Int,
    sy: Int,
    sx: Int,
    fh: Int,
    fw: Int,
    tolerance: Float64,
) abi("C") -> Int:
    var image = fp(image_addr)
    var result = bp(result_addr)
    var stack = ip(stack_addr)
    var footprint = bp(footprint_addr)
    var target = image[sy * w + sx]
    var top = 1
    var found = 0
    stack[0] = Int64(sy * w + sx)
    result[sy * w + sx] = 1
    var ry = fh // 2
    var rx = fw // 2
    while top > 0:
        top -= 1
        var pos = Int(stack[top])
        var y = pos // w
        var x = pos - y * w
        found += 1
        for ky in range(fh):
            for kx in range(fw):
                if footprint[ky * fw + kx] == 0 or (ky == ry and kx == rx):
                    continue
                var yy = y + ky - ry
                var xx = x + kx - rx
                if yy < 0 or yy >= h or xx < 0 or xx >= w:
                    continue
                var q = yy * w + xx
                if result[q] != 0:
                    continue
                if abs(image[q] - target) <= tolerance:
                    result[q] = 1
                    stack[top] = Int64(q)
                    top += 1
    return found


def boundary_at(
    labels: IPtr,
    y: Int,
    x: Int,
    h: Int,
    w: Int,
    connectivity: Int,
    boundary_mode: Int,
    background: Int64,
) -> UInt8:
    var center = labels[y * w + x]
    var mark = False
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            if dy == 0 and dx == 0:
                continue
            if connectivity == 1 and abs(dy) + abs(dx) != 1:
                continue
            var yy = y + dy
            var xx = x + dx
            if yy < 0 or yy >= h or xx < 0 or xx >= w:
                continue
            var other = labels[yy * w + xx]
            if other == center:
                continue
            if boundary_mode == 0:
                mark = True
            elif boundary_mode == 1:
                if center != background:
                    mark = True
            else:
                if center == background or other != background:
                    mark = True
    return UInt8(1) if mark else UInt8(0)


@export("msi_find_boundaries")
def msi_find_boundaries(
    labels_addr: Int,
    result_addr: Int,
    h: Int,
    w: Int,
    connectivity: Int,
    boundary_mode: Int,
    background: Int64,
) abi("C"):
    var labels = ip(labels_addr)
    var result = bp(result_addr)
    comptime W = simd_width_of[DType.int64]()

    @parameter
    def row(y: Int):
        if w == 1:
            result[y * w] = boundary_at(
                labels, y, 0, h, w, connectivity, boundary_mode, background
            )
            return
        result[y * w] = boundary_at(
            labels, y, 0, h, w, connectivity, boundary_mode, background
        )
        var x = 1
        while x + W <= w - 1:
            var center = labels.load[width=W](y * w + x)
            var left = labels.load[width=W](y * w + x - 1)
            var right = labels.load[width=W](y * w + x + 1)
            var mark = center.ne(left) | center.ne(right)
            if y > 0:
                var upper = labels.load[width=W]((y - 1) * w + x)
                if boundary_mode == 2:
                    mark |= center.ne(upper) & (
                        center.eq(background) | upper.ne(background)
                    )
                else:
                    mark |= center.ne(upper)
                if connectivity == 2:
                    var upper_left = labels.load[width=W](
                        (y - 1) * w + x - 1
                    )
                    var upper_right = labels.load[width=W](
                        (y - 1) * w + x + 1
                    )
                    if boundary_mode == 2:
                        mark |= center.ne(upper_left) & (
                            center.eq(background) | upper_left.ne(background)
                        )
                        mark |= center.ne(upper_right) & (
                            center.eq(background) | upper_right.ne(background)
                        )
                    else:
                        mark |= center.ne(upper_left) | center.ne(upper_right)
            if y + 1 < h:
                var lower = labels.load[width=W]((y + 1) * w + x)
                if boundary_mode == 2:
                    mark |= center.ne(lower) & (
                        center.eq(background) | lower.ne(background)
                    )
                else:
                    mark |= center.ne(lower)
                if connectivity == 2:
                    var lower_left = labels.load[width=W](
                        (y + 1) * w + x - 1
                    )
                    var lower_right = labels.load[width=W](
                        (y + 1) * w + x + 1
                    )
                    if boundary_mode == 2:
                        mark |= center.ne(lower_left) & (
                            center.eq(background) | lower_left.ne(background)
                        )
                        mark |= center.ne(lower_right) & (
                            center.eq(background) | lower_right.ne(background)
                        )
                    else:
                        mark |= center.ne(lower_left) | center.ne(lower_right)
            if boundary_mode == 1:
                mark &= center.ne(background)
            elif boundary_mode == 2:
                mark = (
                    center.ne(left)
                    & (center.eq(background) | left.ne(background))
                ) | (
                    center.ne(right)
                    & (center.eq(background) | right.ne(background))
                ) | mark
            result.store(y * w + x, mark.cast[DType.uint8]())
            x += W
        while x < w - 1:
            result[y * w + x] = boundary_at(
                labels, y, x, h, w, connectivity, boundary_mode, background
            )
            x += 1
        result[y * w + w - 1] = boundary_at(
            labels, y, w - 1, h, w, connectivity, boundary_mode, background
        )

    if h * w < PARALLEL_THRESHOLD or h == 1:
        for y in range(h):
            row(y)
    else:
        var tasks = min(h, MAX_ROW_TASKS)

        @parameter
        def rows(task: Int):
            var start = task * h // tasks
            var end = (task + 1) * h // tasks
            for y in range(start, end):
                row(y)

        sync_parallelize[rows](tasks)


@export("msi_remove_small")
def msi_remove_small(
    src_addr: Int,
    dst_addr: Int,
    queue_addr: Int,
    h: Int,
    w: Int,
    min_size: Int,
    connectivity: Int,
) abi("C"):
    var src = bp(src_addr)
    var dst = bp(dst_addr)
    var queue = ip(queue_addr)
    var n = h * w
    for i in range(n):
        dst[i] = src[i]
    for start in range(n):
        if dst[start] == 0 or dst[start] == 2:
            continue
        var head = 0
        var tail = 1
        queue[0] = Int64(start)
        dst[start] = 2
        while head < tail:
            var pos = Int(queue[head])
            head += 1
            var y = pos // w
            var x = pos - y * w
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if dy == 0 and dx == 0:
                        continue
                    if connectivity == 1 and abs(dy) + abs(dx) != 1:
                        continue
                    var yy = y + dy
                    var xx = x + dx
                    if yy < 0 or yy >= h or xx < 0 or xx >= w:
                        continue
                    var q = yy * w + xx
                    if dst[q] == 1:
                        dst[q] = 2
                        queue[tail] = Int64(q)
                        tail += 1
        var value = UInt8(0) if tail < min_size else UInt8(2)
        for j in range(tail):
            dst[Int(queue[j])] = value
    for i in range(n):
        if dst[i] == 2:
            dst[i] = 1


@export("msi_clear_border")
def msi_clear_border(
    labels_addr: Int,
    dst_addr: Int,
    marked_addr: Int,
    h: Int,
    w: Int,
    max_label: Int,
    buffer_size: Int,
    background: Int64,
) abi("C"):
    var labels = ip(labels_addr)
    var dst = ip(dst_addr)
    var marked = bp(marked_addr)
    for i in range(max_label + 1):
        marked[i] = 0
    for y in range(h):
        for x in range(w):
            if y < buffer_size + 1 or y >= h - buffer_size - 1 or x < buffer_size + 1 or x >= w - buffer_size - 1:
                var label = labels[y * w + x]
                if label >= 0 and label <= Int64(max_label):
                    marked[Int(label)] = 1
    for i in range(h * w):
        var label = labels[i]
        if label >= 0 and label <= Int64(max_label) and marked[Int(label)] != 0:
            dst[i] = background
        else:
            dst[i] = label
