import Testing
@testable import gloss_server

@Test func cosineIdentity() {
    let v: [Float] = [1, 2, 3, 4]
    #expect(abs(cosine(v, v) - 1.0) < 1e-9)
}

@Test func matryoshkaIsUnitNorm() {
    let v: [Float] = (0..<1024).map { Float($0 % 7) - 3 }
    let m = matryoshka(v, dim: 256)
    #expect(m.count == 256)
    var n: Float = 0; for x in m { n += x * x }
    #expect(abs(n.squareRoot() - 1.0) < 1e-5)
}
