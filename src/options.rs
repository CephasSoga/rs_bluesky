use std::sync::Arc;
use std::fmt::Display;

pub enum FetchType {
    Bluesky,
    Unknown,
}
impl Display for  FetchType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::Bluesky => "Bluesky",
            _ => "Unknown",
        };
        write!(f, "{}", name)
    }
}

impl FetchType {
    pub fn from(value: Arc<serde_json::Value>) -> FetchType {

        let value = Arc::try_unwrap(value).unwrap_or_else(|v| (*v).clone());
        match value["function"].as_str() {
            Some("bluesky") => FetchType::Bluesky,
            _ => FetchType::Unknown,
        }
    
    }

    pub fn from_str(s: &str) -> FetchType {
        match s {
            "bluesky" => FetchType::Bluesky,
            _ => FetchType::Unknown,
        }
    }
}