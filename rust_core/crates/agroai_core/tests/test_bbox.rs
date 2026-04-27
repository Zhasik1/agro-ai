use agroai_core::bbox::{bbox_area_value, bbox_iou_value, nms_indices, BBox};
use ndarray::array;
use proptest::prelude::*;

fn bbox_strategy() -> impl Strategy<Value = BBox> {
    (0f32..200f32, 0f32..200f32, 1f32..50f32, 1f32..50f32).prop_map(|(x, y, w, h)| BBox {
        x1: x,
        y1: y,
        x2: x + w,
        y2: y + h,
    })
}

#[test]
fn area_is_non_negative() {
    let b = BBox {
        x1: 10.0,
        y1: 10.0,
        x2: 20.0,
        y2: 30.0,
    };
    assert_eq!(bbox_area_value(b), 200.0);
}

#[test]
fn nms_keeps_top_scored_non_overlapping() {
    let boxes = array![
        [0.0, 0.0, 10.0, 10.0],
        [1.0, 1.0, 11.0, 11.0],
        [20.0, 20.0, 30.0, 30.0],
    ];
    let scores = vec![0.9f32, 0.8f32, 0.95f32];

    let keep = nms_indices(boxes.view(), &scores, 0.5).expect("nms should succeed");
    assert_eq!(keep, vec![2, 0]);
}

proptest! {
    #[test]
    fn iou_is_symmetric_and_bounded(a in bbox_strategy(), b in bbox_strategy()) {
        let iou_ab = bbox_iou_value(a, b);
        let iou_ba = bbox_iou_value(b, a);

        prop_assert!((0.0..=1.0).contains(&iou_ab));
        prop_assert!((iou_ab - iou_ba).abs() <= 1e-6);
    }
}
