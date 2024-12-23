use serde::{Serialize, Deserialize};

#[derive(Deserialize)]
pub struct Event {
    pub time_us: i64,
    pub did: String,
    pub commit: Option<Commit>,
}

#[derive(Deserialize)]
pub struct Commit {
    pub operation: String,
    pub collection: String,
    pub record: Option<serde_json::Value>,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct FeedPost {
    pub text: String,
}