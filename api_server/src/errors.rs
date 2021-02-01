use no_captcha::errors::Error as NoCaptchaError;
use serde::Serialize;
use serde_derive::Serialize;
use std::io::Error as IOError;

#[derive(Debug, Serialize)]
#[serde(tag = "err", content = "meta")]
#[serde(rename_all = "snake_case")]
pub enum Error {
    InvalidRecognitionRequest,
    Generic(String),

    #[serde(skip)]
    IOError(IOError),
    #[serde(skip)]
    NoCAPTCHA(NoCaptchaError),
}

impl Error {
    pub fn msg<S>(source: S) -> Error
    where
        S: Into<String>,
    {
        Error::Generic(source.into())
    }
}

impl From<NoCaptchaError> for Error {
    fn from(error: NoCaptchaError) -> Error {
        Error::NoCAPTCHA(error)
    }
}

impl From<IOError> for Error {
    fn from(error: IOError) -> Error {
        Error::IOError(error)
    }
}

impl tide::IntoResponse for Error {
    fn into_response(self) -> tide::Response {
        tide::Response::new(500)
            .set_header("Content-Type", "application/json")
            .body_json(&self)
            .unwrap()
    }
}

pub type Result<T, E = Error> = std::result::Result<T, E>;
#[derive(Serialize)]
pub struct Response<T>(Result<T>)
where
    T: Serialize;

impl<T> From<Result<T>> for Response<T>
where
    T: Serialize,
{
    fn from(result: Result<T>) -> Response<T>
    where
        T: Serialize,
    {
        Response(result)
    }
}

impl<T> tide::IntoResponse for Response<T>
where
    T: Serialize + Send,
{
    fn into_response(self) -> tide::Response {
        if let Err(e) = self.0 {
            e.into_response()
        } else {
            tide::Response::new(200)
                .set_header("Content-Type", "application/json")
                .body_json(&self)
                .unwrap()
        }
    }
}
