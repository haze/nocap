use async_std::task;
use tide::Request;
use no_captcha::{CaptchaRegistry, CaptchaChallenge};
use serde_derive::{Serialize, Deserialize};

mod errors;
use errors::Error;

/// RecaptchaRequest represents the main ways of consuming the API
/// 1. Base64 Image upload
#[derive(Serialize, Deserialize, Debug)]
struct RecognitionRequest {
    challenge: CaptchaChallenge,

    #[serde(flatten)]
    image: Image,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "image_type", content = "image")]
#[serde(rename_all = "snake_case")]
enum Image {
    Base64(String),
    Bytes(Vec<u8>),
}

async fn handle_raw_image_upload(mut req: Request<CaptchaRegistry>) -> errors::Response<no_captcha::Prediction> {
    Ok(match req.body_json::<RecognitionRequest>().await {
        Ok(RecognitionRequest { image: Image::Base64(data), challenge }) => {
            match base64::decode(&data) {
                Ok(decoded_base64) => {
                    let input_str = unsafe { String::from_utf8_unchecked(decoded_base64) };
                    match req.state().predict(&challenge, input_str) {
                        Ok(prediction) => prediction,
                        Err(_) => return Err(Error::msg("Prediction failed")).into(),
                    }
                }
                Err(_) => return Err(Error::msg("Invalid image Base64")).into(),
            }
        },
        Err(err) => {
            dbg!(&err);
            return Err(Error::InvalidRecognitionRequest).into();
        }
        _ => unimplemented!(),
    }).into()
}

async fn async_main() -> errors::Result<()> {
    let registry = CaptchaRegistry::load_from_models_dir("../models/")?;
    let mut app = tide::with_state(registry);
    app.at("/recognize").post(handle_raw_image_upload);
    app.listen("127.0.0.1:5000").await?;
    Ok(())
}

fn main() -> errors::Result<()> {
    task::block_on(async_main())
}
