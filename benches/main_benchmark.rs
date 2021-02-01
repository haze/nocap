use criterion::{criterion_group, criterion_main, Criterion};
use no_captcha::CaptchaRegistry;

pub fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("Load All Models", |b| {
        b.iter(|| CaptchaRegistry::load_from_models_dir("models/").expect("This should not fail"))
    });
}

criterion_group!(
    name = benches;
    config = Criterion::default().sample_size(10);
    targets = criterion_benchmark
);
criterion_main!(benches);
