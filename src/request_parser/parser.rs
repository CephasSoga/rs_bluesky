


use crate::request_parser::params::*;
use serde::{Deserialize, Serialize};
use std::net::IpAddr;

#[derive(Debug, Deserialize)]
struct RawCallRequest {
    caller: RawCaller,
    target: String,
    args: RawArgs,
}

#[derive(Debug, Deserialize, Serialize)]
struct RawCaller {
    id: String,
    ipaddr: IpAddr,
    queue: i32,
    status: i64,
    mode: String,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum RawArgs {
    Database {
        function: String,
        count: String,
        uri: String,
        user: Option<String>,
        pwd: Option<String>,
        document: Option<serde_json::Map<String, serde_json::Value>>,
    },
    Task {
        function: String,
        count: String,
        look_for: serde_json::Value,
        params: Option<serde_json::Map<String, serde_json::Value>>,
    },
}

pub struct CallParser;

impl CallParser {
    pub fn fast_parse_json(query: &str) -> Result<CallRequest, String> {
        let raw: RawCallRequest = serde_json::from_str(query).map_err(|e| e.to_string())?;

        let caller = Caller {
            id: raw.caller.id,
            ipaddr: raw.caller.ipaddr,
            queue: raw.caller.queue,
            status: Status::from_int(raw.caller.status),
            mode: Mode::from_str(&raw.caller.mode),
        };

        let target = TargetService::from_str(&raw.target);

        let args = match (&target, raw.args) {
            (TargetService::Database, RawArgs::Database {
                function, count, uri, user, pwd, document
            }) => Args {
                for_database: Some(DatabaseArgs {
                    function: DatabaseFunction::from_str(&function),
                    count: ObjectCount::from_str(&count),
                    uri,
                    user,
                    pwd,
                    document: document.map(|d| d.into_iter().collect()),
                }),
                for_task: None,
            },
            (TargetService::Task, RawArgs::Task {
                function, count, look_for, params
            }) => Args {
                for_database: None,
                for_task: Some(TaskArgs {
                    function: TaskFunction::from_str(&function),
                    count: TaskCount::from_str(&count),
                    look_for: LookFor::from_json(&look_for),
                    params: params.map(|p| p.into_iter().collect()),
                }),
            },
            _ => return Err("Target and args mismatch".to_string()),
        };

        Ok(CallRequest {
            caller,
            target,
            args,
        })
    }
}
