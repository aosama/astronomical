mod config;
mod model_discovery;
mod performance_attribution;
mod retired_memory_policy;

fn write_config(home_directory: &std::path::Path, config_json: &str) {
    let config_directory = home_directory.join(".astronomical");
    std::fs::create_dir_all(&config_directory).expect("config directory should be created");
    std::fs::write(config_directory.join("config.json"), config_json)
        .expect("config file should be written");
}
