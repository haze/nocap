use std::{io::Error as IOError, sync::PoisonError};
use strum::ParseError;

#[derive(Debug)]
pub enum Error {
    IOError(IOError),
    TensorflowError(tensorflow::Code),
    ModelLoad(crate::CaptchaChallenge),
    StrumParseError(ParseError),
    MutexError,
}

impl From<ParseError> for Error {
    fn from(error: ParseError) -> Error {
        Error::StrumParseError(error)
    }
}

impl<T> From<PoisonError<T>> for Error {
    fn from(_: PoisonError<T>) -> Error {
        Error::MutexError
    }
}

impl From<IOError> for Error {
    fn from(source: IOError) -> Error {
        Error::IOError(source)
    }
}

impl From<tensorflow::Status> for Error {
    fn from(status: tensorflow::Status) -> Error {
        Error::TensorflowError(status.code())
    }
}

pub type Result<T, E = Error> = std::result::Result<T, E>;
