import Foundation

/// The image model's input from a picture, its person matte and a head box, as
/// `hairdeck.infer.image.Pipeline.prep` and `matte_crop` make it — with Pillow's own integer
/// arithmetic, so the pixels are the reference's:
///   1. the picture with the matte as alpha, composited on white (`Image.alpha_composite`);
///   2. the box (rounded half to even) cut out of it, transparent outside the picture, and resized to
///      res × res with Pillow's bicubic filter on premultiplied alpha (`Image.resize` of an RGBA image);
///   3. composited on white again.
/// The matte crop for the colour model is the matte cut by the same box and resized with nearest
/// neighbour, thresholded at 127.
///
/// The head box is the union of the hair and face boxes with an 8 % margin, made square
/// (`headBox`); where the two boxes come from is not this file's concern.
public enum ModelInput {
    /// `crop_box` at inference: the square box around hair ∪ face with the margin on every side.
    public static func headBox(hair: [Double]?, face: [Double]?, margin: Double = 0.08) -> [Double]? {
        guard let b = hair.map({ h in face.map { f in [min(h[0], f[0]), min(h[1], f[1]), max(h[2], f[2]), max(h[3], f[3])] } ?? h }) ?? face else { return nil }
        let cx = (b[0] + b[2]) / 2, cy = (b[1] + b[3]) / 2
        let side = max(b[2] - b[0], b[3] - b[1]) * (1 + 2 * margin)
        return [cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2]
    }

    @inline(__always) static func shiftDiv255(_ a: UInt32) -> UInt32 { ((a >> 8) &+ a) >> 8 }
    @inline(__always) static func mulDiv255(_ a: UInt32, _ b: UInt32) -> UInt8 { let t = a * b + 128; return UInt8(((t >> 8) + t) >> 8) }

    /// Pillow's `alpha_composite` of an RGBA pixel over opaque white → RGB.
    @inline(__always) static func overWhite(_ r: UInt8, _ g: UInt8, _ b: UInt8, _ a: UInt8) -> (UInt8, UInt8, UInt8) {
        if a == 0 { return (255, 255, 255) }
        let sa = UInt32(a)
        let outa255 = sa * 255 + 255 * (255 - sa)
        let coef1 = sa * 255 * 255 * (1 << 7) / outa255
        let coef2 = 255 * (1 << 7) - coef1
        func ch(_ s: UInt8) -> UInt8 { UInt8(shiftDiv255(UInt32(s) * coef1 + 255 * coef2 + (0x80 << 7)) >> 7) }
        return (ch(r), ch(g), ch(b))
    }

    static func bicubic(_ x0: Double) -> Double {
        let a = -0.5
        let x = abs(x0)
        if x < 1 { return ((a + 2) * x - (a + 3)) * x * x + 1 }
        if x < 2 { return (((x - 5) * x + 8) * x - 4) * a }
        return 0
    }

    /// Pillow's resampling coefficients for one axis (`precompute_coeffs`, bicubic, 8-bit fixed point).
    static func coefficients(_ inSize: Int, _ outSize: Int) -> (bounds: [(Int, Int)], kk: [[Int32]]) {
        let scale = Double(inSize) / Double(outSize)
        let filterscale = max(scale, 1)
        let support = 2.0 * filterscale
        let ss = 1.0 / filterscale
        var bounds: [(Int, Int)] = [], kk: [[Int32]] = []
        for xx in 0..<outSize {
            let center = (Double(xx) + 0.5) * scale
            var xmin = Int(center - support + 0.5)
            if xmin < 0 { xmin = 0 }
            var xmax = Int(center + support + 0.5)
            if xmax > inSize { xmax = inSize }
            xmax -= xmin
            var k = [Double](repeating: 0, count: xmax)
            var ww = 0.0
            for x in 0..<xmax {
                let w = bicubic((Double(x + xmin) - center + 0.5) * ss)
                k[x] = w
                ww += w
            }
            if ww != 0 { for x in 0..<xmax { k[x] /= ww } }
            bounds.append((xmin, xmax))
            kk.append(k.map { $0 < 0 ? Int32(-0.5 + $0 * Double(1 << 22)) : Int32(0.5 + $0 * Double(1 << 22)) })
        }
        return (bounds, kk)
    }

    @inline(__always) static func clip8(_ v: Int32) -> UInt8 { UInt8(max(0, min(255, v >> 22))) }

    /// Pillow's bicubic resize of 8-bit channels (horizontal pass, then vertical): [h × w × c] → [oh × ow × c].
    static func resize(_ src: [UInt8], _ w: Int, _ h: Int, _ c: Int, to ow: Int, _ oh: Int) -> [UInt8] {
        var cur = src, cw = w
        if ow != w {
            let (bounds, kk) = coefficients(w, ow)
            var out = [UInt8](repeating: 0, count: h * ow * c)
            for y in 0..<h {
                for x in 0..<ow {
                    let (xmin, n) = bounds[x]
                    for ch in 0..<c {
                        var ss: Int32 = 1 << 21
                        for i in 0..<n { ss &+= Int32(cur[(y * w + xmin + i) * c + ch]) &* kk[x][i] }
                        out[(y * ow + x) * c + ch] = clip8(ss)
                    }
                }
            }
            cur = out
            cw = ow
        }
        if oh != h {
            let (bounds, kk) = coefficients(h, oh)
            var out = [UInt8](repeating: 0, count: oh * cw * c)
            for y in 0..<oh {
                let (ymin, n) = bounds[y]
                for x in 0..<cw {
                    for ch in 0..<c {
                        var ss: Int32 = 1 << 21
                        for i in 0..<n { ss &+= Int32(cur[((ymin + i) * cw + x) * c + ch]) &* kk[y][i] }
                        out[(y * cw + x) * c + ch] = clip8(ss)
                    }
                }
            }
            cur = out
        }
        return cur
    }

    static func ints(_ box: [Double]) -> (Int, Int, Int, Int) {
        let r = box.map { Int($0.rounded(.toNearestOrEven)) }
        return (r[0], r[1], r[2], r[3])
    }

    /// rgb (h × w × 3) and its person matte (h × w, 0..255; an empty matte, max ≤ 127, keeps the whole
    /// picture) → the model input, res × res × 3.
    public static func make(rgb: [UInt8], matte: [UInt8], width w: Int, height h: Int, box: [Double], res: Int = 448) -> [UInt8] {
        let keepAll = (matte.max() ?? 0) <= 127
        let (x0, y0, x1, y1) = ints(box)
        let cw = x1 - x0, chh = y1 - y0
        // the crop of the white composite as premultiplied RGBA (alpha 255 inside the picture, 0 outside)
        var canvas = [UInt8](repeating: 0, count: cw * chh * 4)
        for y in max(y0, 0)..<min(y1, h) {
            for x in max(x0, 0)..<min(x1, w) {
                let s = y * w + x, d = ((y - y0) * cw + (x - x0)) * 4
                let (r, g, b) = overWhite(rgb[s * 3], rgb[s * 3 + 1], rgb[s * 3 + 2], keepAll ? 255 : matte[s])
                canvas[d] = r; canvas[d + 1] = g; canvas[d + 2] = b; canvas[d + 3] = 255
            }
        }
        let small = resize(canvas, cw, chh, 4, to: res, res)
        var out = [UInt8](repeating: 0, count: res * res * 3)
        for i in 0..<(res * res) {
            let a = small[i * 4 + 3]
            // premultiplied → straight alpha (`rgba2rgbA`), then over white
            var p = (small[i * 4], small[i * 4 + 1], small[i * 4 + 2])
            if a != 255 && a != 0 {
                func un(_ c: UInt8) -> UInt8 { UInt8(min(255, 255 * Int(c) / Int(a))) }
                p = (un(p.0), un(p.1), un(p.2))
            }
            let (r, g, b) = overWhite(p.0, p.1, p.2, a)
            out[i * 3] = r; out[i * 3 + 1] = g; out[i * 3 + 2] = b
        }
        return out
    }

    /// The matte cut by the box and resized with nearest neighbour, thresholded at 127: res × res of 0 / 1
    /// (all 1 for an empty result, as the reference keeps the whole image then).
    public static func matteCrop(matte: [UInt8], width w: Int, height h: Int, box: [Double], res: Int = 448) -> [UInt8] {
        let (x0, y0, x1, y1) = ints(box)
        let cw = x1 - x0, chh = y1 - y0
        var out = [UInt8](repeating: 0, count: res * res)
        for y in 0..<res {
            let sy = Int((Double(y) + 0.5) * Double(chh) / Double(res)) + y0
            for x in 0..<res {
                let sx = Int((Double(x) + 0.5) * Double(cw) / Double(res)) + x0
                if sx >= 0, sx < w, sy >= 0, sy < h, matte[sy * w + sx] > 127 { out[y * res + x] = 1 }
            }
        }
        return out.contains(1) ? out : [UInt8](repeating: 1, count: res * res)
    }
}
