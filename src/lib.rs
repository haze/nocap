use rayon::prelude::*;
use serde_derive::{Deserialize, Serialize};
use std::{collections::HashMap, fs, str::FromStr, sync::Mutex};
use strum::VariantNames;
use strum_macros::{Display, EnumString, EnumVariantNames, IntoStaticStr};
use tensorflow::{Graph, Session, Tensor};

pub mod errors;

fn silence_tensorflow() {
    std::env::set_var("TF_CPP_MIN_LOG_LEVEL", "3");
}

#[deny(
    missing_debug_implementations,
    missing_docs,
    trivial_casts,
    trivial_numeric_casts,
    unused_extern_crates,
    unused_import_braces,
    unused_qualifications,
    unused_results,
    bad_style,
    const_err,
    dead_code,
    improper_ctypes,
    legacy_directory_ownership,
    non_shorthand_field_patterns,
    no_mangle_generic_items,
    overflowing_literals,
    path_statements,
    patterns_in_fns_without_body,
    plugin_as_library,
    private_in_public,
    safe_extern_statics,
    unconditional_recursion,
    unused,
    unused_allocation,
    unused_comparisons,
    unused_parens,
    while_true
)]
#[derive(
    Debug,
    Eq,
    PartialEq,
    Display,
    Hash,
    IntoStaticStr,
    EnumVariantNames,
    EnumString,
    Serialize,
    Deserialize,
)]
#[strum(serialize_all = "snake_case")]
#[allow(missing_docs)]
#[serde(rename_all = "snake_case")]
/// CaptchaChallenge represents all accepted reCaptcha challenge types
pub enum CaptchaChallenge {
    AFireHydrant,
    Bridges,
    Cars,
    Motorcycles,
    PalmTrees,
    Stairs,
    StoreFront,
    Tractors,
    Bicycles,
    Bus,
    Crosswalks,
    MountainsOrHills,
    ParkingMeters,
    Statues,
    Taxis,
    TrafficLights,
}

impl CaptchaChallenge {
    /// is_valid_challenge_os_str is the same as is_valid_challenge but will
    /// return 'false' if the source string cannot be converted to utf8
    fn is_valid_challenge_os_str<S>(name: S) -> bool
    where
        S: AsRef<std::ffi::OsStr>,
    {
        if let Some(string) = name.as_ref().to_str() {
            return Self::is_valid_challenge(string);
        }
        false
    }

    /// is_valid_challenge tests 'name' for its prescence in the EnumVariant
    fn is_valid_challenge<S>(name: S) -> bool
    where
        S: AsRef<str>,
    {
        for var in CaptchaChallenge::VARIANTS {
            if name.as_ref() == *var {
                return true;
            }
        }
        false
    }
}

/// SavedModelMap employs a mutex around Session because running sessions performs interior
/// mutability
type SavedModelMap = HashMap<CaptchaChallenge, Mutex<CaptchaModel>>;

#[derive(Debug)]
pub struct CaptchaModel {
    session: Session,
    graph: Graph,
}

#[derive(Debug)]
pub struct CaptchaRegistry {
    items: SavedModelMap,
}

impl CaptchaRegistry {
    pub fn load_from_models_dir<P>(path: P) -> errors::Result<CaptchaRegistry>
    where
        P: AsRef<std::path::Path>,
    {
        let model_directories: Vec<fs::DirEntry> = {
            let mut vec = Vec::new();
            for dir in path.as_ref().read_dir()? {
                vec.push(dir?);
            }
            vec
        };

        let model_count = model_directories.len();
        silence_tensorflow();
        Ok(CaptchaRegistry {
            items: model_directories
                .into_par_iter()
                .filter(|dir: &fs::DirEntry| {
                    CaptchaChallenge::is_valid_challenge_os_str(dir.file_name())
                })
                .try_fold(
                    || SavedModelMap::new(),
                    |mut acc, dir: fs::DirEntry| {
                        let saved_model_file = dir.path().join("saved_model.pb");
                        let challenge = CaptchaChallenge::from_str(
                            dir.file_name()
                                .to_str()
                                .expect("Could not retrieve Model's name"),
                        )
                        .unwrap();
                        if !saved_model_file.exists() {
                            return Err(errors::Error::ModelLoad(challenge));
                        } else {
                            let mut graph = Graph::new();
                            let session = Session::from_saved_model(
                                &tensorflow::SessionOptions::new(),
                                &["serve"],
                                &mut graph,
                                dir.path(),
                            )?;
                            acc.insert(challenge, Mutex::new(CaptchaModel { session, graph }));
                        }
                        Ok(acc)
                    },
                )
                .try_reduce(
                    || SavedModelMap::with_capacity(model_count),
                    |mut m, t| {
                        for (k, v) in t.into_iter() {
                            m.insert(k, v);
                        }
                        Ok(m)
                    },
                )?,
        })
    }

    pub fn predict(
        &self,
        challenge: &CaptchaChallenge,
        image: String,
    ) -> errors::Result<Prediction> {
        let model = self
            .items
            .get(challenge)
            .expect("This should not happen")
            .lock()?;

        // inptus
        let input_operation = model.graph.operation_by_name_required("Placeholder")?;
        let input_tensor = Tensor::new(&[1u64]).with_values(&[image])?;

        let mut output_step = tensorflow::SessionRunArgs::new();
        output_step.add_feed(&input_operation, 0, &input_tensor);

        let scores_out =
            output_step.request_fetch(&model.graph.operation_by_name_required("scores")?, 0);

        model.session.run(&mut output_step)?;
        let predictions: Tensor<f32> = output_step.fetch(scores_out)?;

        Ok(Prediction {
            affirmative_confidence: predictions[0],
            negative_confidence: predictions[1],
        })
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Prediction {
    affirmative_confidence: f32,
    negative_confidence: f32,
}

impl Prediction {
    // TODO(haze): better signals
    pub fn is_mainly_affirmative(&self) -> bool {
        self.affirmative_confidence >= 0.50 && self.negative_confidence < 0.50
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path;

    #[test]
    fn load_models() -> errors::Result<()> {
        CaptchaRegistry::load_from_models_dir(path::Path::new("models/")).map(|_| ())
    }

    fn load_image_into_string<A>(path: A) -> errors::Result<String>
    where
        A: AsRef<path::Path>,
    {
        Ok(unsafe { String::from_utf8_unchecked(std::fs::read(path)?) })
    }

    #[test]
    fn prediction() -> errors::Result<()> {
        let test_image = load_image_into_string("./bus.png")?;
        let registry: CaptchaRegistry =
            CaptchaRegistry::load_from_models_dir(path::Path::new("models/"))?;
        let prediction = registry.predict(&CaptchaChallenge::Bus, test_image);
        dbg!(&prediction);
        Ok(())
    }

    fn files_in<A>(path: A) -> errors::Result<Vec<fs::DirEntry>>
    where
        A: AsRef<path::Path>,
    {
        path::Path::new(path.as_ref())
            .read_dir()?
            .try_fold(Vec::new(), |mut acc, r| {
                acc.push(r?);

                Ok(acc)
            })
    }

    fn test_challenge_size<A>(path: A, registry: &CaptchaRegistry) -> errors::Result<()>
    where
        A: AsRef<path::Path>,
    {
        let challenges: Vec<fs::DirEntry> = files_in(path)?;
        for challenge in challenges {
            test_challenge(
                challenge.path(),
                challenge
                    .file_name()
                    .to_str()
                    .ok_or_else(|| errors::Error::MutexError)?,
                registry,
            )?;
        }
        Ok(())
    }

    fn test_challenge<A>(path: A, size: &str, registry: &CaptchaRegistry) -> errors::Result<()>
    where
        A: AsRef<path::Path>,
    {
        let challenge = path
            .as_ref()
            .file_name()
            .map(|osstr| osstr.to_str())
            .flatten()
            .expect("Expecting challenge from folder")
            .replace(" ", "_");
        println!("Beginning test on {}/{}...", size, &challenge);
        let challenge = CaptchaChallenge::from_str(&*challenge)?;
        test_matches(&path, registry, &challenge)?;
        test_not_matches(&path, registry, &challenge)?;
        Ok(())
    }

    fn test_matches<A>(
        dir: A,
        registry: &CaptchaRegistry,
        challenge: &CaptchaChallenge,
    ) -> errors::Result<()>
    where
        A: AsRef<path::Path>,
    {
        test_images(dir.as_ref().join("matches/"), registry, challenge, true)
    }

    fn test_not_matches<A>(
        dir: A,
        registry: &CaptchaRegistry,
        challenge: &CaptchaChallenge,
    ) -> errors::Result<()>
    where
        A: AsRef<path::Path>,
    {
        test_images(
            dir.as_ref().join("not matches/"),
            registry,
            challenge,
            false,
        )
    }

    fn test_images<A>(
        dir: A,
        registry: &CaptchaRegistry,
        challenge: &CaptchaChallenge,
        expecting_correct: bool,
    ) -> errors::Result<()>
    where
        A: AsRef<path::Path>,
    {
        let files: Vec<fs::DirEntry> = files_in(dir)?;
        let (mut correct, mut incorrect) = (0.0, 0.0);
        for file in files {
            println!("[{}] {:?}", challenge, &file.path());
            let results: Prediction =
                registry.predict(challenge, load_image_into_string(file.path())?)?;
            if results.is_mainly_affirmative() {
                if expecting_correct {
                    correct += 1.0;
                } else {
                    incorrect += 1.0;
                }
            } else {
                if expecting_correct {
                    incorrect += 1.0;
                } else {
                    correct += 1.0;
                }
            }
            println!("{:?}", results);
        }
        let total: f64 = correct + incorrect;
        println!(
            "[{}] {:.1}% {} Correct, {:.1}% {} Incorrect ({} total)",
            challenge,
            100.0 * (correct / total as f64),
            correct,
            100.0 * (incorrect / total as f64),
            incorrect,
            total,
        );
        Ok(())
    }

    #[test]
    fn models() -> errors::Result<()> {
        let registry: CaptchaRegistry =
            CaptchaRegistry::load_from_models_dir(path::Path::new("models/"))?;
        let sizes: Vec<fs::DirEntry> = files_in("./test_data/")?;
        for size in sizes {
            test_challenge_size(size.path(), &registry)?;
        }
        Ok(())
    }
}
