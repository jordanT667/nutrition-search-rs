//! `nutrisearch`: ask the French dietary guidelines a question, get the pages that answer it.
//!
//!     nutrisearch "combien de poisson par semaine ?"     one question
//!     nutrisearch                                        interactive mode
//!     nutrisearch eval                                   hit@k and MRR on the 40 answerable questions
//!     nutrisearch bench                                  index build time and queries per second
//!
//! Options: -k N (results, default 5), --passages PATH, --questions PATH, --json.

use std::io::{self, BufRead, IsTerminal, Write};
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use nutrisearch::{evaluate, Index, Passage, Question};

const DEFAULT_PASSAGES: &str = "../nutrition-rag/data/processed/passages.json";
const DEFAULT_QUESTIONS: &str = "eval/questions.jsonl";
const KS: [usize; 4] = [1, 3, 5, 8];

struct Style {
    on: bool,
}

impl Style {
    fn paint(&self, code: &str, s: &str) -> String {
        if self.on {
            format!("\x1b[{code}m{s}\x1b[0m")
        } else {
            s.to_string()
        }
    }
    fn bold(&self, s: &str) -> String {
        self.paint("1", s)
    }
    fn dim(&self, s: &str) -> String {
        self.paint("2", s)
    }
    fn red(&self, s: &str) -> String {
        self.paint("1;31", s)
    }
    fn mark(&self, s: &str) -> String {
        self.paint("1;97;41", s)
    }
}

struct Args {
    command: Option<String>,
    query: Vec<String>,
    k: usize,
    passages: PathBuf,
    questions: PathBuf,
    json: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        command: None,
        query: Vec::new(),
        k: 5,
        passages: std::env::var_os("NUTRISEARCH_PASSAGES")
            .map(PathBuf::from)
            .unwrap_or_else(|| DEFAULT_PASSAGES.into()),
        questions: DEFAULT_QUESTIONS.into(),
        json: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-k" => {
                a.k = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .filter(|&k| k > 0)
                    .ok_or("-k needs a positive number")?
            }
            "--passages" => a.passages = it.next().ok_or("--passages needs a path")?.into(),
            "--questions" => a.questions = it.next().ok_or("--questions needs a path")?.into(),
            "--json" => a.json = true,
            "-h" | "--help" => a.command = Some("help".into()),
            "eval" | "bench" if a.command.is_none() && a.query.is_empty() => a.command = Some(arg),
            _ => a.query.push(arg),
        }
    }
    Ok(a)
}

fn load_passages(path: &PathBuf) -> Result<Vec<Passage>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| {
        format!(
            "cannot read {} ({e}).\nBuild it with nutrition-rag (python -m src.nutrirag.ingest), or pass --passages PATH.",
            path.display()
        )
    })?;
    serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))
}

fn load_questions(path: &PathBuf) -> Result<Vec<Question>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("cannot read {} ({e})", path.display()))?;
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).map_err(|e| format!("{}: {e}", path.display())))
        .collect()
}

/// A window of the passage around the first matching word, with matching words highlighted.
fn snippet(index: &Index, text: &str, terms: &[String], width: usize, st: &Style) -> String {
    let words: Vec<&str> = text.split_whitespace().collect();
    let hit = |w: &str| index.tokenizer.tokens(w).iter().any(|t| terms.contains(t));
    let first = words.iter().position(|w| hit(w)).unwrap_or(0);
    let start = first.saturating_sub(6);
    let mut parts: Vec<String> = Vec::new();
    if start > 0 {
        parts.push("…".into());
    }
    let mut len = 0;
    for w in &words[start..] {
        if len + w.chars().count() > width {
            parts.push("…".into());
            break;
        }
        parts.push(if hit(w) { st.mark(w) } else { w.to_string() });
        len += w.chars().count() + 1;
    }
    parts.join(" ")
}

fn print_results(index: &Index, query: &str, k: usize, st: &Style) {
    let terms = index.tokenizer.tokens(query);
    let t0 = Instant::now();
    let hits = index.search(query, k);
    let took = t0.elapsed();
    println!();
    println!("{} {}", st.red("❯"), st.bold(query));
    if terms.is_empty() || hits.first().is_none_or(|h| h.score <= 0.0) {
        println!(
            "  {}",
            st.dim("Aucun passage ne contient ces mots. Reformulez avec des mots du document.")
        );
        return;
    }
    println!(
        "  {}",
        st.dim(&format!(
            "mots cherchés : {}   ·   {} passages en {:.0} µs",
            terms.join(" · "),
            index.passages.len(),
            took.as_secs_f64() * 1e6
        ))
    );
    let best = hits[0].score;
    for (rank, h) in hits.iter().enumerate().filter(|(_, h)| h.score > 0.0) {
        let p = &index.passages[h.index];
        let bar_len = ((h.score / best) * 14.0).round() as usize;
        let bar = format!("{}{}", "█".repeat(bar_len), "░".repeat(14 - bar_len));
        let page = format!("p.{:<3}", p.page);
        let rank_s = format!("{:>2}", rank + 1);
        println!();
        println!(
            "  {}  {}  {} {}",
            if rank == 0 { st.red(&rank_s) } else { st.bold(&rank_s) },
            st.bold(&page),
            if rank == 0 { st.red(&bar) } else { st.dim(&bar) },
            st.dim(&format!("{:.2}", h.score))
        );
        println!("      {}", snippet(index, &p.text, &terms, 96, st));
        let why: Vec<String> = index
            .bm25
            .explain(&terms, h.index)
            .iter()
            .map(|(t, s)| format!("{t} +{s:.2}"))
            .collect();
        println!("      {}", st.dim(&format!("pourquoi : {}", why.join("  "))));
    }
    println!();
}

fn print_json(index: &Index, query: &str, k: usize) {
    let hits: Vec<serde_json::Value> = index
        .search(query, k)
        .iter()
        .map(|h| {
            let p = &index.passages[h.index];
            serde_json::json!({"id": p.id, "page": p.page, "score": h.score, "text": p.text})
        })
        .collect();
    println!("{}", serde_json::json!({"query": query, "hits": hits}));
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    let st = Style {
        on: io::stdout().is_terminal() && std::env::var_os("NO_COLOR").is_none(),
    };
    if args.command.as_deref() == Some("help") {
        println!("{}", include_str!("help.txt"));
        return Ok(());
    }

    let t0 = Instant::now();
    let passages = load_passages(&args.passages)?;
    let index = Index::new(passages);
    let build = t0.elapsed();

    match args.command.as_deref() {
        Some("eval") => {
            let questions = load_questions(&args.questions)?;
            let m = evaluate(&index, &questions, &KS);
            println!(
                "{}",
                st.bold(&format!("BM25 en Rust, {} questions avec réponse", m.questions))
            );
            for (k, v) in &m.hit_at {
                println!("  hit@{k:<2} {v:.3}");
            }
            println!("  MRR    {:.3}", m.mrr);
        }
        Some("bench") => {
            let questions = load_questions(&args.questions)?;
            let rounds = 200;
            let t = Instant::now();
            let mut checksum = 0usize;
            for _ in 0..rounds {
                for q in &questions {
                    checksum += index.search(&q.question, 5)[0].index;
                }
            }
            let n = rounds * questions.len();
            let secs = t.elapsed().as_secs_f64();
            println!("{}", st.bold("Vitesse (build release, un seul cœur)"));
            println!(
                "  index de {} passages construit en {:.2} ms",
                index.passages.len(),
                build.as_secs_f64() * 1e3
            );
            println!(
                "  {n} recherches en {:.3} s : {:.0} requêtes/s, {:.1} µs par requête",
                secs,
                n as f64 / secs,
                secs / n as f64 * 1e6
            );
            std::hint::black_box(checksum);
        }
        _ if !args.query.is_empty() => {
            let q = args.query.join(" ");
            if args.json {
                print_json(&index, &q, args.k)
            } else {
                print_results(&index, &q, args.k, &st)
            }
        }
        _ => {
            println!(
                "{}",
                st.bold("Posez une question sur les recommandations nutritionnelles (ligne vide pour quitter).")
            );
            let stdin = io::stdin();
            loop {
                print!("{} ", st.red("?"));
                io::stdout().flush().ok();
                let mut line = String::new();
                if stdin.lock().read_line(&mut line).map_err(|e| e.to_string())? == 0 || line.trim().is_empty() {
                    break;
                }
                print_results(&index, line.trim(), args.k, &st);
            }
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("nutrisearch: {e}");
            ExitCode::FAILURE
        }
    }
}
