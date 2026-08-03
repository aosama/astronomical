mod config;
mod model_discovery;
mod performance_attribution;
mod retired_memory_policy;

fn write_config(home_directory: &std::path::Path, config_json: &str) {
    let config_directory = home_directory.join(".astronomical");
    std::fs::create_dir_all(&config_directory).expect("config directory should be created");
    let config_json = match serde_json::from_str::<serde_json::Value>(config_json) {
        Ok(mut parsed_config_json) => match parsed_config_json.as_object_mut() {
            Some(config_object)
                if !config_object.contains_key("prefill_chunck_size_optimizer_enabled")
                    && !config_object.contains_key("fixed_prefill_chunck_tokens") =>
            {
                config_object.insert(
                    "prefill_chunck_size_optimizer_enabled".to_owned(),
                    serde_json::Value::Bool(true),
                );
                parsed_config_json.to_string()
            }
            _ => config_json.to_owned(),
        },
        Err(_) => config_json.to_owned(),
    };
    std::fs::write(config_directory.join("config.json"), config_json)
        .expect("config file should be written");
}
