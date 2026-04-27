use agroai_core::image_ops::{
    bgr_to_rgb_array, crop_bbox_array, decode_image_rgb_array, normalize_imagenet_array,
    resize_bilinear_rgb,
};
use image::{DynamicImage, ImageFormat, RgbImage};
use ndarray::array;
use std::io::Cursor;

fn sample_png_bytes() -> Vec<u8> {
    let mut img = RgbImage::new(4, 4);
    for y in 0..4 {
        for x in 0..4 {
            img.put_pixel(x, y, image::Rgb([(x * 40) as u8, (y * 40) as u8, 120]));
        }
    }

    let mut cursor = Cursor::new(Vec::<u8>::new());
    DynamicImage::ImageRgb8(img)
        .write_to(&mut cursor, ImageFormat::Png)
        .expect("encoding should succeed");
    cursor.into_inner()
}

#[test]
fn decode_image_rgb_returns_expected_shape() {
    let bytes = sample_png_bytes();
    let decoded = decode_image_rgb_array(&bytes).expect("decode should succeed");
    assert_eq!(decoded.shape(), &[4, 4, 3]);
}

#[test]
fn resize_bilinear_changes_size() {
    let input = decode_image_rgb_array(&sample_png_bytes()).expect("decode should succeed");
    let resized = resize_bilinear_rgb(input.view(), 8, 8).expect("resize should succeed");
    assert_eq!(resized.shape(), &[8, 8, 3]);
}

#[test]
fn bgr_to_rgb_swaps_channels() {
    let bgr = array![[[10u8, 20u8, 30u8]]];
    let rgb = bgr_to_rgb_array(bgr.view()).expect("conversion should succeed");
    assert_eq!(rgb[[0, 0, 0]], 30);
    assert_eq!(rgb[[0, 0, 1]], 20);
    assert_eq!(rgb[[0, 0, 2]], 10);
}

#[test]
fn normalize_imagenet_outputs_chw() {
    let input = decode_image_rgb_array(&sample_png_bytes()).expect("decode should succeed");
    let normalized = normalize_imagenet_array(input.view()).expect("normalize should succeed");
    assert_eq!(normalized.shape(), &[3, 4, 4]);
}

#[test]
fn crop_bbox_respects_bounds() {
    let input = decode_image_rgb_array(&sample_png_bytes()).expect("decode should succeed");
    let cropped = crop_bbox_array(input.view(), (1, 1, 3, 3)).expect("crop should succeed");
    assert_eq!(cropped.shape(), &[2, 2, 3]);
}
