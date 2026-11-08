//! Runs the Gherkin feature files in `features/` against cudash.
//!
//! `cargo test --test features` runs every scenario this machine can run. Scenarios that need
//! hardware or tools this machine lacks are skipped, and each skip prints its reason.
//!
//! It takes the arguments of a regular test binary, so `just test` and `cargo test -- ...` work:
//! `--test-threads n` (or `RUST_TEST_THREADS`), `--skip s`, `--exact`, and bare words that must
//! appear in `<feature>::<scenario>`. Other flags, such as `--nocapture`, are accepted and ignored.
//! Scenarios run one at a time unless `--test-threads` says otherwise, because each creates its own
//! GPU context and buffers.

mod steps;

use cucumber::World as _;
use steps::World;

#[tokio::main]
async fn main() {
    let mut threads = std::env::var("RUST_TEST_THREADS")
        .ok()
        .and_then(|n| n.parse().ok());
    let (mut include, mut skip, mut exact) = (Vec::new(), Vec::new(), false);
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.split_once('=') {
            Some(("--test-threads", n)) => threads = n.parse().ok(),
            Some(("--skip", s)) => skip.push(s.to_owned()),
            _ if arg == "--test-threads" => threads = args.next().and_then(|n| n.parse().ok()),
            _ if arg == "--skip" => skip.extend(args.next()),
            _ if arg == "--exact" => exact = true,
            _ if !arg.starts_with('-') => include.push(arg),
            _ => {}
        }
    }
    let matches = move |name: &str, f: &String| {
        if exact {
            name == f
        } else {
            name.contains(f.as_str())
        }
    };

    // `with_default_cli` stops cucumber parsing the arguments itself, which would reject these.
    World::cucumber()
        .with_default_cli()
        .max_concurrent_scenarios(threads.filter(|&n| n > 0).unwrap_or(1))
        .filter_run_and_exit("features", move |feature, rule, scenario| {
            let name = format!("{}::{}", feature.name, scenario.name);
            (include.is_empty() || include.iter().any(|f| matches(&name, f)))
                && !skip.iter().any(|f| matches(&name, f))
                && steps::selected(feature, rule, scenario)
        })
        .await;
}
