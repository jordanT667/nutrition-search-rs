//! The Rust scores must equal the Python ones (rank_bm25), query by query and passage by passage.

use std::path::Path;

use nutrisearch::{Index, Passage};
use serde::Deserialize;

#[derive(Deserialize)]
struct Expected {
    query: String,
    tokens: Vec<String>,
    scores: Vec<f64>,
}

#[derive(Deserialize)]
struct Small {
    passages: Vec<Passage>,
    queries: Vec<Expected>,
}

fn check(index: &Index, expected: &[Expected]) {
    for e in expected {
        assert_eq!(index.tokenizer.tokens(&e.query), e.tokens, "tokens of {:?}", e.query);
        let got = index.bm25.scores(&e.tokens);
        assert_eq!(got.len(), e.scores.len());
        for (i, (g, w)) in got.iter().zip(&e.scores).enumerate() {
            assert!(
                (g - w).abs() <= 1e-9 * w.abs().max(1.0),
                "{:?}, passage {i}: rust {g} vs python {w}",
                e.query
            );
        }
    }
}

#[test]
fn scores_match_python_on_the_small_corpus() {
    let small: Small = serde_json::from_str(include_str!("fixtures/small.json")).unwrap();
    check(&Index::new(small.passages), &small.queries);
}

/// Needs the real corpus and `python tools/reference.py`; skipped otherwise (e.g. in CI).
#[test]
fn scores_match_python_on_the_real_corpus() {
    let passages = Path::new("../nutrition-rag/data/processed/passages.json");
    let expected = Path::new("target/python_scores.json");
    if !passages.exists() || !expected.exists() {
        eprintln!("skipped: real corpus or Python scores not found");
        return;
    }
    let passages: Vec<Passage> = serde_json::from_str(&std::fs::read_to_string(passages).unwrap()).unwrap();
    let expected: Vec<Expected> = serde_json::from_str(&std::fs::read_to_string(expected).unwrap()).unwrap();
    assert_eq!(expected.len(), 50);
    check(&Index::new(passages), &expected);
}
