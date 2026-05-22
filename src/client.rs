use anyhow::{bail, Context, Result};
use reqwest::blocking::Client as HttpClient;
use reqwest::{Method, StatusCode};
use serde_json::{json, Value};
use std::time::Duration;

#[derive(Clone)]
pub struct Client {
    http: HttpClient,
    endpoint: String,
    user: String,
    password: String,
}

impl Client {
    pub fn new(endpoint: &str, user: &str, password: &str) -> Result<Self> {
        let endpoint = endpoint.trim_end_matches('/').to_string();
        let http = HttpClient::builder()
            .timeout(Duration::from_secs(600))
            .pool_max_idle_per_host(32)
            .build()?;
        Ok(Self {
            http,
            endpoint,
            user: user.into(),
            password: password.into(),
        })
    }

    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    pub fn user(&self) -> &str {
        &self.user
    }

    pub fn password(&self) -> &str {
        &self.password
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.endpoint, path)
    }

    fn request(&self, method: Method, path: &str, body: Option<&Value>) -> Result<Value> {
        let url = self.url(path);
        let mut req = self
            .http
            .request(method.clone(), &url)
            .basic_auth(&self.user, Some(&self.password));
        if let Some(b) = body {
            req = req.json(b);
        }
        let resp = req.send().with_context(|| format!("{} {}", method, url))?;
        let status = resp.status();
        let text = resp
            .text()
            .with_context(|| format!("reading body from {} {}", method, url))?;
        let value: Value = if text.is_empty() {
            Value::Null
        } else {
            serde_json::from_str(&text)
                .with_context(|| format!("decoding JSON from {} {}: {}", method, url, text))?
        };
        if !status.is_success() {
            bail!("{} {} -> {}: {}", method, url, status, value);
        }
        Ok(value)
    }

    pub fn drop_database_if_exists(&self, db: &str) -> Result<()> {
        let url = self.url(&format!("/_db/_system/_api/database/{}", db));
        let resp = self
            .http
            .delete(&url)
            .basic_auth(&self.user, Some(&self.password))
            .send()?;
        if resp.status() == StatusCode::NOT_FOUND {
            return Ok(());
        }
        if !resp.status().is_success() {
            bail!("DELETE database {}: {}", db, resp.status());
        }
        Ok(())
    }

    pub fn create_database(&self, db: &str) -> Result<()> {
        let body = json!({ "name": db });
        self.request(Method::POST, "/_db/_system/_api/database", Some(&body))?;
        Ok(())
    }

    pub fn database_exists(&self, db: &str) -> Result<bool> {
        let url = self.url(&format!("/_db/{}/_api/database/current", db));
        let resp = self
            .http
            .get(&url)
            .basic_auth(&self.user, Some(&self.password))
            .send()?;
        Ok(resp.status().is_success())
    }

    pub fn create_collection(&self, db: &str, name: &str, number_of_shards: u64) -> Result<()> {
        let path = format!("/_db/{}/_api/collection", db);
        let body = json!({ "name": name, "numberOfShards": number_of_shards });
        self.request(Method::POST, &path, Some(&body))?;
        Ok(())
    }

    pub fn collection_count(&self, db: &str, coll: &str) -> Result<u64> {
        let path = format!("/_db/{}/_api/collection/{}/count", db, coll);
        let v = self.request(Method::GET, &path, None)?;
        v["count"]
            .as_u64()
            .context("missing 'count' in collection response")
    }

    pub fn insert_docs(&self, db: &str, coll: &str, docs: &Value) -> Result<()> {
        let path = format!("/_db/{}/_api/document/{}?silent=true", db, coll);
        self.request(Method::POST, &path, Some(docs))?;
        Ok(())
    }

    pub fn create_vector_index(&self, db: &str, coll: &str, def: &Value) -> Result<Value> {
        let path = format!("/_db/{}/_api/index?collection={}", db, coll);
        self.request(Method::POST, &path, Some(def))
    }

    pub fn list_indexes(&self, db: &str, coll: &str, with_hidden: bool) -> Result<Value> {
        let path = format!(
            "/_db/{}/_api/index?collection={}&withHidden={}",
            db, coll, with_hidden
        );
        self.request(Method::GET, &path, None)
    }

    pub fn aql(&self, db: &str, query: &str, bind_vars: Value) -> Result<Vec<Value>> {
        let path = format!("/_db/{}/_api/cursor", db);
        let body = json!({
            "query": query,
            "bindVars": bind_vars,
            "batchSize": 10_000,
        });
        let mut v = self.request(Method::POST, &path, Some(&body))?;
        let mut out: Vec<Value> = v["result"]
            .as_array()
            .context("missing 'result' in cursor response")?
            .clone();
        while v["hasMore"].as_bool().unwrap_or(false) {
            let id = v["id"]
                .as_str()
                .context("missing cursor 'id' for continuation")?
                .to_string();
            let next_path = format!("/_db/{}/_api/cursor/{}", db, id);
            v = self.request(Method::POST, &next_path, None)?;
            out.extend(
                v["result"]
                    .as_array()
                    .context("missing 'result' in cursor continuation")?
                    .clone(),
            );
        }
        Ok(out)
    }
}
