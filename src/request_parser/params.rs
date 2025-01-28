//! This module defines various data structures and enumerations used for parsing and handling
//! request parameters in the `aggr_market_data` application. The primary structures include
//! `Caller`, `TaskArgs`, `DatabaseArgs`, and `CallRequest`, each of which encapsulates specific
//! details about the request and its context.
//!
//! # Structures
//!
//! - `Caller`: Represents the entity making the request, including its ID, IP address, queue status,
//!   and mode of operation.
//!
//! - `TaskArgs`: Contains details about a task to be performed, including the function type, count,
//!   and optional parameters.
//!
//! - `DatabaseArgs`: Encapsulates information for database operations, such as the function type,
//!   object count, URI, user credentials, and the document to be manipulated.
//!
//! - `CallRequest`: The main structure that ties together the caller information, target service,
//!   and the specific arguments for the request.
//!
//! # Enumerations
//!
//! - `Status`: Represents the status of a request, with possible values being `Pending`, `Finished`,
//!   and `Failed`.
//!
//! - `Mode`: Defines the mode of operation for a request, such as `Async`, `Sync`, `Batch`, `Stream`,
//!   `None`, and `Unknown`.
//!
//! - `TaskFunction`: Enumerates the different functions that can be performed in a task, including
//!   `AggregatedPolling`, `RealTimeMarketData`, `RealTimeBlueSky`, `RealTimeSocialMedia`, `WebSearch`,
//!   `ChatGPT`, and `NLP`.
//!
//! - `TaskCount`: Specifies the count type for tasks, such as `Single`, `Multiple`, `Batch`, `Stream`,
//!   `None`, and `Unknown`.
//!
//! - `DatabaseFunction`: Lists the possible database operations, including `Read`, `Insert`, `Update`,
//!   `Replace`, and `Delete`.
//!
//! - `ObjectCount`: Indicates whether the database operation involves a single object or multiple objects.
//!
//! - `TargetService`: Identifies the target service for the request, either `Database` or `Task`.
//!
//! - `Args`: A wrapper enumeration that can hold either `DatabaseArgs` or `TaskArgs`.
//!
//! This module leverages the `serde` crate for serialization and deserialization of the defined
//! structures and enumerations, facilitating easy conversion to and from JSON format.

use std::{collections::HashMap, net::IpAddr};
use serde_json::Value;
use serde::{Serialize, Deserialize};


// ************* Caller *************** | START
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Status {
    Pending,
    Finished,
    Failed,
}
impl Status {
    pub fn from_int(i: i64) -> Self {
        match i {
            0 => Status::Pending,
            1 => Status::Finished,
            2 => Status::Failed,
            _ => Status::Failed,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Mode {
    Async,
    Sync,
    Batch,
    Stream,
    None,
    Unknown,
}

impl Mode {
    pub fn from_str(s: &str) -> Self {
        match s {
            "async" => Mode::Async,
            "sync" => Mode::Sync,
            "batch" => Mode::Batch,
            "stream" => Mode::Stream,
            "none" => Mode::None,
            _ => Mode::Unknown,
        }
    }

    pub fn to_str(&self) -> &str {
        match self {
            Mode::Async => "async",
            Mode::Sync => "sync",
            Mode::Batch => "batch",
            Mode::Stream => "stream",
            Mode::None => "none",
            Mode::Unknown => "unknown",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Caller {
    pub id: String,
    pub ipaddr: IpAddr,
    pub queue: i32,
    pub status: Status,
    pub mode: Mode
}
// ************* Caller *************** | END

// ************* TaskFunction *************** | START
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskFunction {
    AggregatedPolling,
    RealTimeMarketData,
    RealTimeBlueSky,
    RealTimeSocialMedia,
    WebSearch,
    ChatGPT,
    NLP,
    Unknown
}
impl TaskFunction {
    pub fn from_str(s: &str) -> Self {
        match s {
            "aggregated_polling" => TaskFunction::AggregatedPolling,
            "real_time_market_data" => TaskFunction::RealTimeMarketData,
            "real_time_blue_sky" => TaskFunction::RealTimeBlueSky,
            "real_time_social_media" => TaskFunction::RealTimeSocialMedia,
            "web_search" => TaskFunction::WebSearch,
            "chat_gpt" => TaskFunction::ChatGPT,
            "nlp" => TaskFunction::NLP,
            _ => TaskFunction::Unknown,
        }
    }

    pub fn to_str(&self) -> &str {
        match self {
            TaskFunction::AggregatedPolling => "aggregated_polling",
            TaskFunction::RealTimeMarketData => "real_time_market_data",
            TaskFunction::RealTimeBlueSky => "real_time_blue_sky",
            TaskFunction::RealTimeSocialMedia => "real_time_social_media",
            TaskFunction::WebSearch => "web_search",
            TaskFunction::ChatGPT => "chat_gpt",
            TaskFunction::NLP => "nlp",
            TaskFunction::Unknown => "unknown",
        }
    }  
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskCount {
    Single,
    Multiple,
    Batch,
    Stream,
    None,
    Unknown
}
impl TaskCount {
    pub fn from_str(s: &str) -> Self {
        match s {
            "single" => TaskCount::Single,
            "multiple" => TaskCount::Multiple,
            "batch" => TaskCount::Batch,
            "stream" => TaskCount::Stream,
            "none" => TaskCount::None,
            _ => TaskCount::Unknown,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LookFor {
    pub where_: String,
}
impl LookFor {
    pub fn from_str(s: &str) -> Self {
        LookFor {
            where_: s.to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskArgs {
    pub function: TaskFunction,
    pub count: TaskCount,
    pub look_for: LookFor,
    pub params: Option<HashMap<String, Value>>
} 
// ************* TaskFunction *************** | END

// ************* Database *************** | START
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DatabaseFunction {
    Read,
    Insert,
    Update,
    Replace,
    Delete,
}
impl DatabaseFunction {
    pub fn default() -> Self {
        DatabaseFunction::Read
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "read" => DatabaseFunction::Read,
            "insert" => DatabaseFunction::Insert,
            "update" => DatabaseFunction::Update,
            "replace" => DatabaseFunction::Replace,
            "delete" => DatabaseFunction::Delete,
            _ => DatabaseFunction::Read,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ObjectCount {
    One,
    Many
}
impl ObjectCount {
    pub fn default() -> Self {
        ObjectCount::One
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "one" => ObjectCount::One,
            "many" => ObjectCount::Many,
            _ => ObjectCount::One,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseArgs {
    pub function: DatabaseFunction,
    pub count: ObjectCount,
    pub uri: String,
    pub user: Option<String>,
    pub pwd: Option<String>,
    pub document: Option<HashMap<String, Value>>
}
impl DatabaseArgs {
    
}
// ************* Database *************** | END

// ************* ReqParams *************** | START
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TargetService {
    Database,
    Task,
    Unknown,
}
impl TargetService {
    pub fn from_str(s: &str) -> Self {
        match s {
            "database" => TargetService::Database,
            "task" => TargetService::Task,
            _ => TargetService::Unknown,
        }
    }
    
    pub fn to_str(&self) -> &str {
        match self {
            TargetService::Database => "database",
            TargetService::Task => "task",
            TargetService::Unknown => "unknown",
        }
    }
    
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Args {
    pub for_database: Option<DatabaseArgs>,
    pub for_task: Option<TaskArgs>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallRequest {
    pub caller: Caller,
    pub target: TargetService,
    pub args: Args,
}
// ************* ReqParams *************** | END
