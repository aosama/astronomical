//! Serialization contract for the image generation response DTO, moved
//! beside its type from the integration lane (issue #1610).

use crate::{OpenAiGeneratedImageParts, OpenAiImageGenerationResponse};

#[test]
fn should_serialize_one_generated_image_with_reproducibility_metadata() {
    let response = OpenAiImageGenerationResponse::new(
        1_787_010_400,
        OpenAiGeneratedImageParts {
            b64_json: "iVBORw0KGgoAAAANSUhEUg==".to_owned(),
            mime_type: "image/png".to_owned(),
            model_revision: "0123456789abcdef".to_owned(),
            effective_seed: u64::MAX,
            width: 1_024,
            height: 768,
        },
    );

    assert_eq!(
        serde_json::to_string(&response).expect("the generated image response should serialize"),
        r#"{"created":1787010400,"data":[{"b64_json":"iVBORw0KGgoAAAANSUhEUg==","mime_type":"image/png","model_revision":"0123456789abcdef","seed":18446744073709551615,"width":1024,"height":768}]}"#
    );
}
