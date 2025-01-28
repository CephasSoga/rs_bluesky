use crate::request_parser::params::*;
use serde_json::Value;
//use std::collections::HashMap;
use std::net::IpAddr;

pub struct CallParser;
impl CallParser {
    pub fn default_parse_json(query_string: &str) -> CallRequest {
        serde_json::from_str(query_string).unwrap()
    }

    pub fn key_lookup_parse_json(query_string: &str) -> Result<CallRequest, String> {
        let json_value: Value = serde_json::from_str(query_string).map_err(|e| e.to_string())?;
        
        let caller = Self::parse_caller(&json_value)?;
        let target = Self::parse_target_service(&json_value)?;
        let args = Self::parse_args(&json_value, &target)?;

        Ok(CallRequest {
            caller,
            target,
            args,
        })
    }

    fn parse_caller(json_value: &Value) -> Result<Caller, String> {
        let caller_obj = json_value.get("caller").ok_or("Missing 'caller' field")?;
        let id = caller_obj.get("id").and_then(Value::as_str).ok_or("Missing 'id' field")?.to_string();
        let ipaddr = caller_obj.get("ipaddr").and_then(Value::as_str).ok_or("Missing 'ipaddr' field")?.parse::<IpAddr>().map_err(|e| e.to_string())?;
        let queue = caller_obj.get("queue").and_then(Value::as_i64).ok_or("Missing 'queue' field")? as i32;
        let status = caller_obj.get("status").and_then(Value::as_i64).map(Status::from_int).ok_or("Missing 'status' field")?;
        let mode = caller_obj.get("mode").and_then(Value::as_str).map(Mode::from_str).ok_or("Missing 'mode' field")?;

        Ok(Caller {
            id,
            ipaddr,
            queue,
            status,
            mode,
        })
    }

    fn parse_target_service(json_value: &Value) -> Result<TargetService, String> {
        let target_str = json_value.get("target").and_then(Value::as_str).ok_or("Missing 'target' field")?;
        Ok(TargetService::from_str(target_str))
    }

    fn parse_args(json_value: &Value, target: &TargetService) -> Result<Args, String> {
        match target {
            TargetService::Database => {
                let db_args = json_value.get("args").ok_or("Missing 'args' field")?;
                let function = db_args.get("function").and_then(Value::as_str).map(DatabaseFunction::from_str).ok_or("Missing 'function' field")?;
                let count = db_args.get("count").and_then(Value::as_str).map(ObjectCount::from_str).ok_or("Missing 'count' field")?;
                let uri = db_args.get("uri").and_then(Value::as_str).ok_or("Missing 'uri' field")?.to_string();
                let user = db_args.get("user").and_then(Value::as_str).map(String::from);
                let pwd = db_args.get("pwd").and_then(Value::as_str).map(String::from);
                let document = db_args.get("document").and_then(Value::as_object).map(|doc| doc.clone().into_iter().collect());

                Ok(Args {
                    for_database: Some(DatabaseArgs {
                        function,
                        count,
                        uri,
                        user,
                        pwd,
                        document,
                    }),
                    for_task: None,
                })
            }
            TargetService::Task => {
                let task_args = json_value.get("args").ok_or("Missing 'args' field")?;
                let function = task_args.get("function").and_then(Value::as_str).map(TaskFunction::from_str).ok_or("Missing 'function' field")?;
                let count = task_args.get("count").and_then(Value::as_str).map(TaskCount::from_str).ok_or("Missing 'count' field")?;
                let look_for = task_args.get("look_for").ok_or("Missing 'look_for' field".to_string()).and_then(|v| serde_json::from_value(v.clone()).map_err(|e| format!("Invalid 'look_for' field: {}", e)))?;
                let params = task_args.get("params").and_then(Value::as_object).map(|p| p.clone().into_iter().collect());

                Ok(Args {
                    for_database: None,
                    for_task: Some(TaskArgs {
                        function,
                        count,
                        look_for,
                        params,
                    }),
                })
            }
            TargetService::Unknown => Err("Unknown target service".to_string()),
        }
    }
}