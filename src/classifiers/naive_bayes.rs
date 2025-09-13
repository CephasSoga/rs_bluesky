//! Naive Bayes classifier (cheap-first, batch-friendly second).
//!
//! Modes:
//! - Sparse mode (default): fast & cheap for single texts (uses HashMaps).
//! - Dense mode (optional): matrix dot product for big batches.
//!
//! Author: Cephas & ChatGPT

use std::collections::HashMap;
use std::fs;
use serde::{Deserialize, Serialize};
use serde_json;
use regex::Regex;
use thiserror::Error;
use ndarray::{Array2, Array1, Axis};

/// Possible errors in Naive Bayes
#[derive(Error, Debug)]
pub enum NaiveBayesError {
    #[error("Failed to save/load: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

/// The Naive Bayes model.
/// Stores both HashMaps (cheap sparse mode) and an optional dense weight matrix.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NaiveBayes {
    /// P(class)
    class_probabilities: HashMap<String, f64>,
    /// P(word|class)
    word_probabilities: HashMap<String, HashMap<String, f64>>,
    /// Vocabulary → index (used for dense mode)
    vocab: HashMap<String, usize>,

    /// Optional dense weights (class × vocab)
    /// Only built if you call `build_dense_matrix()`
    #[serde(skip)]
    wmat: Option<Array2<f32>>,
    /// Class order for mapping rows of `wmat` back to labels
    #[serde(skip)]
    pub class_labels: Vec<String>,
}

impl NaiveBayes {
    /// Load NB model from a JSON file (previously trained & saved).
    pub fn load_from_file(path: &str) -> Result<Self, NaiveBayesError> {
        let content = fs::read_to_string(path)?;
        let mut model: NaiveBayes = serde_json::from_str(&content)?;
        // Build vocab from word_probabilities (cheap init)
        let mut vocab = HashMap::new();
        let mut idx = 0;
        for class_probs in model.word_probabilities.values() {
            for word in class_probs.keys() {
                if !vocab.contains_key(word) {
                    vocab.insert(word.clone(), idx);
                    idx += 1;
                }
            }
        }
        model.vocab = vocab;
        Ok(model)
    }

    /// Save the model to disk
    pub fn save_to_file(&self, path: &str) -> Result<(), NaiveBayesError> {
        let content = serde_json::to_string_pretty(self)?;
        fs::write(path, content)?;
        Ok(())
    }

    /// Build a dense weight matrix `W` for fast batched inference.
    /// Shape: (num_classes, vocab_size).
    pub fn build_dense_matrix(&mut self) {
    let num_classes = self.class_probabilities.len();
    let vocab_size = self.vocab.len();

    let mut wmat = Array2::<f32>::zeros((num_classes, vocab_size));
    let mut class_labels = Vec::new();

    for (class_idx, (class, _)) in self.class_probabilities.iter().enumerate() {
        class_labels.push(class.clone());

        if let Some(word_probs) = self.word_probabilities.get(class) {
            for (word, &p) in word_probs {
                if let Some(&col) = self.vocab.get(word) {
                    // Store log-probabilities
                    wmat[(class_idx, col)] = p.ln() as f32;
                }
            }
        }
    }

    self.wmat = Some(wmat);
    self.class_labels = class_labels;
}


    // ------------------------
    // Sparse (cheap) inference
    // ------------------------

    /// Classify a single text (cheap sparse mode).
    pub fn classify_sparse(&self, text: &str) -> Option<String> {
        let re = Regex::new(r"\w+").unwrap();
        let tokens: Vec<String> = re
            .find_iter(&text.to_lowercase())
            .map(|m| m.as_str().to_string())
            .collect();

        let mut best_class: Option<String> = None;
        let mut best_score = f64::NEG_INFINITY;

        for (class, &class_prob) in &self.class_probabilities {
            let mut log_prob = class_prob.ln();

            if let Some(word_probs) = self.word_probabilities.get(class) {
                for token in &tokens {
                    let p = word_probs.get(token).copied().unwrap_or(1.0 / self.vocab.len() as f64);
                    log_prob += p.ln();
                }
            }
            if log_prob > best_score {
                best_score = log_prob;
                best_class = Some(class.clone());
            }
        }

        best_class
    }

    // ------------------------
    // Dense (batch) inference
    // ------------------------

    /// Convert texts into a bag-of-words matrix (X).
    /// Each row = one text, each column = word count.
    pub fn texts_to_matrix(&self, texts: &[String]) -> Array2<f32> {
        let mut x = Array2::<f32>::zeros((texts.len(), self.vocab.len()));
        let re = Regex::new(r"\w+").unwrap();

        for (i, text) in texts.iter().enumerate() {
            for token in re.find_iter(&text.to_lowercase()) {
                if let Some(&j) = self.vocab.get(token.as_str()) {
                    x[(i, j)] += 1.0;
                }
            }
        }
        x
    }

    /// Classify a batch of texts using dense matrix multiplication.
    pub fn classify_batch_dense(&self, texts: &[String]) -> Vec<String> {
        let wmat = match &self.wmat {
            Some(w) => w,
            None => panic!("Dense matrix not built! Call build_dense_matrix() first."),
        };

        let x = self.texts_to_matrix(texts);
        let log_class_prior: Array1<f32> = self.class_labels.iter()
            .map(|c| self.class_probabilities[c].ln() as f32)
            .collect::<Array1<f32>>()
            .into();
        
        //let scores = x.dot(&wmat.t()); // (batch_size × num_classes)
        let scores = x.dot(&wmat.t()) + &log_class_prior;

        let mut results = Vec::new();
        for row in scores.axis_iter(Axis(0)) {
            let (best_idx, _) = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap();
            results.push(self.class_labels[best_idx].clone());
        }
        results
    }
}
