use agroai_core::hashing::{blake3_hex, phash_array, sha256_hex};
use ndarray::Array3;

#[test]
fn sha256_known_vector() {
    let digest = sha256_hex(b"abc").expect("sha256 should succeed");
    assert_eq!(
        digest,
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
}

#[test]
fn blake3_hex_has_expected_length() {
    let digest = blake3_hex(b"agroai").expect("blake3 should succeed");
    assert_eq!(digest.len(), 64);
}

#[test]
fn phash_is_deterministic() {
    let mut img = Array3::<u8>::zeros((32, 32, 3));
    for y in 0..32usize {
        for x in 0..32usize {
            img[(y, x, 0)] = (x * 8) as u8;
            img[(y, x, 1)] = (y * 8) as u8;
            img[(y, x, 2)] = 128;
        }
    }

    let h1 = phash_array(img.view()).expect("phash should succeed");
    let h2 = phash_array(img.view()).expect("phash should succeed");
    assert_eq!(h1, h2);
    assert_eq!(h1.len(), 16);
}
