//! Behaviour of the search and of the evaluation on a tiny corpus, independent of Python.

use nutrisearch::{evaluate, Index, Passage, Question, Tokenizer};

fn passage(page: u32, text: &str) -> Passage {
    Passage {
        id: format!("p{page}"),
        doc: "test".into(),
        page,
        text: text.into(),
    }
}

fn index() -> Index {
    Index::new(vec![
        passage(
            34,
            "Deux fois par semaine du poisson, dont un poisson gras (sardines, maquereau, hareng, saumon).",
        ),
        passage(
            36,
            "Limiter la charcuterie à 150 g par semaine, environ 3 tranches de jambon blanc.",
        ),
        passage(
            35,
            "L'eau est la seule boisson recommandée. Les sodas, même light, doivent être limités.",
        ),
    ])
}

fn question(id: &str, text: &str, pages: &[u32]) -> Question {
    Question {
        id: id.into(),
        kind: "test".into(),
        question: text.into(),
        pages: pages.to_vec(),
    }
}

#[test]
fn search_returns_at_most_k_hits_best_first() {
    let idx = index();
    let hits = idx.search("poisson gras", 2);
    assert_eq!(hits.len(), 2);
    assert_eq!(idx.passages[hits[0].index].page, 34);
    assert!(hits[0].score >= hits[1].score);
}

#[test]
fn k_larger_than_the_corpus_returns_every_passage() {
    assert_eq!(index().search("poisson", 10).len(), 3);
}

#[test]
fn a_query_with_no_known_word_keeps_document_order() {
    let idx = index();
    let hits = idx.search("chocolat", 3);
    assert!(hits.iter().all(|h| h.score == 0.0));
    assert_eq!(hits.iter().map(|h| h.index).collect::<Vec<_>>(), vec![0, 1, 2]);
}

#[test]
fn a_repeated_query_word_counts_twice_as_in_python() {
    let idx = index();
    let once = idx.bm25.scores(&idx.tokenizer.tokens("poisson"));
    let twice = idx.bm25.scores(&idx.tokenizer.tokens("poisson poisson"));
    for (a, b) in once.iter().zip(&twice) {
        assert!((2.0 * a - b).abs() < 1e-12);
    }
}

#[test]
fn an_unknown_word_does_not_change_the_scores() {
    let idx = index();
    let base = idx.bm25.scores(&idx.tokenizer.tokens("charcuterie"));
    let extra = idx.bm25.scores(&idx.tokenizer.tokens("charcuterie chocolat"));
    assert_eq!(base, extra);
}

#[test]
fn explain_lists_only_the_matched_words_largest_first() {
    let idx = index();
    let q = idx.tokenizer.tokens("poisson sardines chocolat");
    let why = idx.bm25.explain(&q, 0);
    let words: Vec<&str> = why.iter().map(|(w, _)| w.as_str()).collect();
    assert_eq!(words.len(), 2);
    assert!(words.contains(&"poisson") && words.contains(&"sardine"));
    assert!(why[0].1 >= why[1].1);
}

#[test]
fn short_words_keep_their_final_s_and_numbers_are_kept() {
    let t = Tokenizer::default();
    // Stemming only applies to words of 5 letters or more: "pois" stays, "tranches" loses its s.
    assert_eq!(t.tokens("pois tranches 150 g"), vec!["pois", "tranche", "150"]);
}

#[test]
fn evaluation_skips_unanswerable_questions_and_ranks_the_first_right_page() {
    let idx = index();
    let questions = vec![
        question("q1", "Combien de poisson par semaine ?", &[34]),
        question("q2", "Quelle boisson recommandée ?", &[35]),
        question("q3", "Combien de charcuterie ?", &[99]),
        question("q4", "Quel est le prix du chocolat ?", &[]),
    ];
    let m = evaluate(&idx, &questions, &[1, 3]);
    // q4 has no answer in the document and is left out; q3's page is never retrieved.
    assert_eq!(m.questions, 3);
    assert_eq!(m.hit_at, vec![(1, 2.0 / 3.0), (3, 2.0 / 3.0)]);
    assert!((m.mrr - 2.0 / 3.0).abs() < 1e-12);
}
