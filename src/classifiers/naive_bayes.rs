use std::collections::HashMap;
use regex::Regex;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use std::io;
use serde_json;
use indicatif::{ProgressBar, ProgressStyle};

use crate::generic_types::{Commit, FeedPost};


#[derive(Error, Debug)]
pub enum NaiveBayesError {
    #[error("Failed to save the model: {0}")]
    SaveError(io::Error),

    #[error("Failed to load the model: {0}")]
    LoadError(io::Error),

    #[error("Failed to serialize or deserialize the model: {0}")]
    SerializationError(serde_json::Error),

    #[error("Invalid input: {0}")]
    InputError(String),
}


#[derive(Debug, Clone, Deserialize, Serialize)]
/// Struct for the Naive Bayes Classifier
pub struct NaiveBayes {
    class_probabilities: HashMap<String, f64>,
    word_probabilities: HashMap<String, HashMap<String, f64>>,
    vocab: HashMap<String, usize>,
}
impl NaiveBayes {
    /// Create a new Naive Bayes classifier
    pub fn new() -> Self {
        NaiveBayes {
            class_probabilities: HashMap::new(),
            word_probabilities: HashMap::new(),
            vocab: HashMap::new(),
        }
    }

    /// Train the classifier with labeled data
    pub fn train(&mut self, data: Vec<(String, String)>) {
        let mut class_counts = HashMap::new();
        let mut word_counts = HashMap::new();

        // Tokenizer regex
        let re = Regex::new(r"\w+").unwrap();

        // Initialize progress bar
        let pb = ProgressBar::new(data.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}")
                .unwrap()
                .progress_chars("#>-"),
        );
        pb.set_message("Training in progress...");


        // Count classes and words
        for (label, text) in data {
            *class_counts.entry(label.clone()).or_insert(0) += 1;

            let words = re
                .find_iter(&text.to_lowercase())
                .map(|m| m.as_str().to_string())
                .collect::<Vec<String>>();

            for word in words {
                self.vocab.entry(word.clone()).or_insert(0);
                *word_counts
                    .entry(label.clone())
                    .or_insert_with(HashMap::new)
                    .entry(word)
                    .or_insert(0) += 1;
            }
            // Increment progress bar
            pb.inc(1);
        }

        // Finish progress bar
        pb.finish_with_message("Training complete!");

        // Calculate class probabilities
        let total_docs = class_counts.values().sum::<usize>() as f64;
        for (label, count) in class_counts {
            self.class_probabilities.insert(label.clone(), count as f64 / total_docs);

            // Calculate word probabilities for each class
            let mut probabilities = HashMap::new();
            let total_words = word_counts
                .get(&label)
                .unwrap_or(&HashMap::new())
                .values()
                .sum::<usize>() as f64;
            let vocab_size = self.vocab.len() as f64;

            for (word, &count) in word_counts.get(&label).unwrap_or(&HashMap::new()) {
                probabilities.insert(
                    word.clone(),
                    (count as f64 + 1.0) / (total_words + vocab_size),
                );
            }

            self.word_probabilities.insert(label.clone(), probabilities);
        }
    }

    /// Classify a given text
    pub fn classify(&self, text: &str) -> Option<String> {
        let re = Regex::new(r"\w+").unwrap();
        let words = re
            .find_iter(&text.to_lowercase())
            .map(|m| m.as_str().to_string())
            .collect::<Vec<String>>();

        let mut scores = HashMap::new();

        for (class, &class_prob) in &self.class_probabilities {
            let mut log_prob = class_prob.ln();

            if let Some(word_probs) = self.word_probabilities.get(class) {
                for word in &words {
                    let word_prob = word_probs.get(word).copied().unwrap_or(1.0 / self.vocab.len() as f64);
                    log_prob += word_prob.ln();
                }
            }

            scores.insert(class.clone(), log_prob);
        }

        scores.into_iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).map(|(label, _)| label)
    }

    /// Save the model to a file
    pub fn save_to_file(&self, path: &str) -> Result<(), NaiveBayesError> {
        let serialized = serde_json::to_string(self).map_err(NaiveBayesError::SerializationError)?;
        std::fs::write(path, serialized).map_err(NaiveBayesError::SaveError)?;
        Ok(())
    }

    /// Load the model from a file
    pub fn load_from_file(path: &str) -> Result<Self, NaiveBayesError> {
        let serialized = std::fs::read_to_string(path).map_err(NaiveBayesError::LoadError)?;
        let model = serde_json::from_str(&serialized).map_err(NaiveBayesError::SerializationError)?;
        Ok(model)
    }
}

pub fn example(commit: Commit) {
    let mut naive_bayes = NaiveBayes::new();

    // Train with sample data
    let training_data = vec![
        ("economy".to_string(), "The economy is improving.".to_string()),
        ("stock_market".to_string(), "Stock prices are rising.".to_string()),
        ("trading".to_string(), "Trading volumes are high today.".to_string()),
    ];
    naive_bayes.train(training_data);
    // Save the model
    naive_bayes.save_to_file("naive_bayes_model.json").unwrap();

    // Load the model
    let loaded_model = NaiveBayes::load_from_file("naive_bayes_model.json").unwrap();

    // Classify incoming messages
    if let Some(record) = commit.record {
        if let Ok(post) = serde_json::from_value::<FeedPost>(record) {
            if let Some(label) = loaded_model.classify(&post.text) {
                if label == "economy" || label == "stock_market" || label == "trading" {
                    println!("Label: {}, Filtered Post: {}", label, post.text);
                }
            }
        }
    }
}