//! BM25 search over the passages of the French dietary guidelines.
//!
//! A from-scratch port of the retriever in `nutrition-rag` (Python, `rank_bm25.BM25Okapi`):
//! same text folding, same tokens, same formula and constants, so the scores match to
//! floating-point precision (checked in `tests/parity.rs`).

use std::collections::HashMap;

use serde::Deserialize;
use unicode_normalization::UnicodeNormalization;

/// One passage of the source document, as written by `nutrition-rag` (`passages.json`).
#[derive(Debug, Clone, Deserialize)]
pub struct Passage {
    pub id: String,
    pub doc: String,
    pub page: u32,
    pub text: String,
}

/// One evaluation question (`eval/questions.jsonl`). `pages` is empty when the
/// document does not answer it.
#[derive(Debug, Clone, Deserialize)]
pub struct Question {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub question: String,
    pub pages: Vec<u32>,
}

const STOPWORDS: &str = "a au aux avec ce ces cet cette dans de des du elle en est et il ils je la le les leur lui \
    ma mais me mes moi mon ne nos notre nous on ou par pas pour qu que qui sa se ses son sur ta te \
    tes toi ton tu un une vos votre vous y d l s c j n m t qu est sont faut peut combien quel quelle \
    quels quelles comment pourquoi quand faire il elle plus";

/// Lowercase, spell out the French ligatures, and drop accents (NFKD, ASCII only).
pub fn fold(text: &str) -> String {
    let lower = text.to_lowercase().replace('œ', "oe").replace('æ', "ae");
    lower.nfkd().filter(char::is_ascii).collect()
}

/// Tokenizer with the same rules as the Python one: `[a-z0-9]+` runs of the folded text,
/// French stopwords and one-letter words removed, light plural stemming on words of 5+ letters.
pub struct Tokenizer {
    stopwords: std::collections::HashSet<&'static str>,
}

impl Default for Tokenizer {
    fn default() -> Self {
        Self {
            stopwords: STOPWORDS.split_whitespace().collect(),
        }
    }
}

impl Tokenizer {
    pub fn tokens(&self, text: &str) -> Vec<String> {
        fold(text)
            .split(|c: char| !(c.is_ascii_lowercase() || c.is_ascii_digit()))
            .filter(|w| w.len() >= 2 && !self.stopwords.contains(w))
            .map(|w| {
                if w.len() > 4 && (w.ends_with('s') || w.ends_with('x')) {
                    w[..w.len() - 1].to_string()
                } else {
                    w.to_string()
                }
            })
            .collect()
    }
}

/// Okapi BM25 with the `rank_bm25` conventions: k1 = 1.5, b = 0.75, and negative IDFs
/// replaced by `epsilon` times the average IDF.
pub struct Bm25 {
    pub k1: f64,
    pub b: f64,
    doc_freqs: Vec<HashMap<String, u32>>,
    doc_len: Vec<f64>,
    avgdl: f64,
    idf: HashMap<String, f64>,
}

impl Bm25 {
    pub fn new(corpus: &[Vec<String>]) -> Self {
        Self::with_params(corpus, 1.5, 0.75, 0.25)
    }

    pub fn with_params(corpus: &[Vec<String>], k1: f64, b: f64, epsilon: f64) -> Self {
        assert!(!corpus.is_empty(), "BM25 needs at least one document");
        let mut doc_freqs = Vec::with_capacity(corpus.len());
        let mut doc_len = Vec::with_capacity(corpus.len());
        let mut df: HashMap<String, u32> = HashMap::new();
        let mut total = 0usize;
        for doc in corpus {
            let mut freqs: HashMap<String, u32> = HashMap::new();
            for t in doc {
                *freqs.entry(t.clone()).or_default() += 1;
            }
            for t in freqs.keys() {
                *df.entry(t.clone()).or_default() += 1;
            }
            total += doc.len();
            doc_len.push(doc.len() as f64);
            doc_freqs.push(freqs);
        }
        let n = corpus.len() as f64;
        let mut idf: HashMap<String, f64> = HashMap::with_capacity(df.len());
        let mut sum = 0.0;
        let mut negative = Vec::new();
        for (t, &f) in &df {
            let f = f as f64;
            let v = (n - f + 0.5).ln() - (f + 0.5).ln();
            sum += v;
            if v < 0.0 {
                negative.push(t.clone());
            }
            idf.insert(t.clone(), v);
        }
        let floor = epsilon * sum / idf.len() as f64;
        for t in negative {
            idf.insert(t, floor);
        }
        Self {
            k1,
            b,
            doc_freqs,
            doc_len,
            avgdl: total as f64 / n,
            idf,
        }
    }

    pub fn len(&self) -> usize {
        self.doc_len.len()
    }

    pub fn is_empty(&self) -> bool {
        self.doc_len.is_empty()
    }

    pub fn idf(&self, term: &str) -> f64 {
        self.idf.get(term).copied().unwrap_or(0.0)
    }

    /// Score of every document for the query tokens (a repeated token counts again, as in Python).
    pub fn scores(&self, query: &[String]) -> Vec<f64> {
        let mut out = vec![0.0; self.len()];
        for q in query {
            let idf = self.idf(q);
            if idf == 0.0 {
                continue;
            }
            for (i, freqs) in self.doc_freqs.iter().enumerate() {
                let tf = freqs.get(q).copied().unwrap_or(0) as f64;
                let norm = self.k1 * (1.0 - self.b + self.b * self.doc_len[i] / self.avgdl);
                out[i] += idf * (tf * (self.k1 + 1.0) / (tf + norm));
            }
        }
        out
    }

    /// Contribution of each query term to one document's score, largest first.
    pub fn explain(&self, query: &[String], doc: usize) -> Vec<(String, f64)> {
        let mut seen: HashMap<&str, f64> = HashMap::new();
        let norm = self.k1 * (1.0 - self.b + self.b * self.doc_len[doc] / self.avgdl);
        for q in query {
            let tf = self.doc_freqs[doc].get(q).copied().unwrap_or(0) as f64;
            if tf > 0.0 {
                *seen.entry(q.as_str()).or_default() += self.idf(q) * (tf * (self.k1 + 1.0) / (tf + norm));
            }
        }
        let mut v: Vec<(String, f64)> = seen.into_iter().map(|(t, s)| (t.to_string(), s)).collect();
        v.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        v
    }
}

/// A search hit: index of the passage and its score.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hit {
    pub index: usize,
    pub score: f64,
}

/// Tokenizer + BM25 index over a list of passages.
pub struct Index {
    pub passages: Vec<Passage>,
    pub tokenizer: Tokenizer,
    pub bm25: Bm25,
}

impl Index {
    pub fn new(passages: Vec<Passage>) -> Self {
        let tokenizer = Tokenizer::default();
        let corpus: Vec<Vec<String>> = passages.iter().map(|p| tokenizer.tokens(&p.text)).collect();
        let bm25 = Bm25::new(&corpus);
        Self {
            passages,
            tokenizer,
            bm25,
        }
    }

    /// The k best passages, highest score first (ties broken by position in the document).
    pub fn search(&self, query: &str, k: usize) -> Vec<Hit> {
        let scores = self.bm25.scores(&self.tokenizer.tokens(query));
        let mut order: Vec<usize> = (0..scores.len()).collect();
        order.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
        order
            .into_iter()
            .take(k)
            .map(|i| Hit {
                index: i,
                score: scores[i],
            })
            .collect()
    }
}

/// Retrieval metrics on the answerable questions, as in `nutrition-rag/src/nutrirag/eval_retrieval.py`.
#[derive(Debug, Clone, PartialEq)]
pub struct Metrics {
    pub questions: usize,
    pub hit_at: Vec<(usize, f64)>,
    pub mrr: f64,
}

pub fn evaluate(index: &Index, questions: &[Question], ks: &[usize]) -> Metrics {
    let max_k = ks.iter().copied().max().unwrap_or(1);
    let mut hits = vec![0usize; ks.len()];
    let mut rr = 0.0;
    let mut n = 0usize;
    for q in questions.iter().filter(|q| !q.pages.is_empty()) {
        n += 1;
        let pages: Vec<u32> = index
            .search(&q.question, max_k)
            .iter()
            .map(|h| index.passages[h.index].page)
            .collect();
        if let Some(rank) = pages.iter().position(|p| q.pages.contains(p)).map(|i| i + 1) {
            rr += 1.0 / rank as f64;
            for (j, &k) in ks.iter().enumerate() {
                if rank <= k {
                    hits[j] += 1;
                }
            }
        }
    }
    let n_f = n.max(1) as f64;
    Metrics {
        questions: n,
        hit_at: ks.iter().zip(&hits).map(|(&k, &h)| (k, h as f64 / n_f)).collect(),
        mrr: rr / n_f,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fold_keeps_ligatures_and_drops_accents() {
        assert_eq!(fold("Œufs, Bœuf et Crème BRÛLÉE"), "oeufs, boeuf et creme brulee");
    }

    #[test]
    fn tokens_remove_stopwords_and_plural() {
        let t = Tokenizer::default();
        assert_eq!(t.tokens("Combien de légumes par jour ?"), vec!["legume", "jour"]);
        assert_eq!(t.tokens("Les noix et les gâteaux"), vec!["noix", "gateau"]);
    }

    #[test]
    fn negative_idf_is_floored() {
        let corpus: Vec<Vec<String>> = [
            ["eau", "sel"],
            ["eau", "sucre"],
            ["eau", "lait"],
            ["eau", "pain"],
            ["eau", "riz"],
        ]
        .iter()
        .map(|d| d.iter().map(|s| s.to_string()).collect())
        .collect();
        let bm = Bm25::new(&corpus);
        // "eau" is in every document: its raw IDF is negative, so it gets epsilon * mean IDF.
        assert!(bm.idf("eau") > 0.0 && bm.idf("eau") < bm.idf("sel"));
    }

    #[test]
    fn explain_adds_up_to_the_score() {
        let passages = vec![
            Passage {
                id: "a".into(),
                doc: "d".into(),
                page: 1,
                text: "poisson gras sardines poisson".into(),
            },
            Passage {
                id: "b".into(),
                doc: "d".into(),
                page: 2,
                text: "jambon charcuterie".into(),
            },
            Passage {
                id: "c".into(),
                doc: "d".into(),
                page: 3,
                text: "eau boisson".into(),
            },
        ];
        let idx = Index::new(passages);
        let q = idx.tokenizer.tokens("poisson sardines");
        let total: f64 = idx.bm25.explain(&q, 0).iter().map(|(_, s)| s).sum();
        assert!((total - idx.bm25.scores(&q)[0]).abs() < 1e-12);
    }
}
