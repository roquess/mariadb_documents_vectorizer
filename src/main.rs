use anyhow::Result;
use clap::{Parser, Subcommand};
use sqlx::mysql::MySqlPool;
use std::path::{Path, PathBuf};
use tracing::info;
use pdf_extract::extract_text;
use sqlx::Row;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    Index {
        #[arg(short, long)]
        file: PathBuf,
    },
    Search {
        #[arg(short, long)]
        query: String,
        #[arg(short, long, default_value = "5")]
        limit: i32,
    },
}

#[derive(Debug)]
pub struct Document {
    name: String,
    description: String,
    embedding: Vec<f32>,
}

#[derive(Debug)]
pub struct SearchResult {
    pub name: String,
    pub description: String,
    pub similarity: f32,
}

pub struct App {
    pool: MySqlPool,
}

impl App {
    pub async fn new(database_url: &str) -> Result<Self> {
        let pool = MySqlPool::connect(database_url).await?;
        let app = Self { pool };
        app.init_database().await?;
        Ok(app)
    }

    async fn init_database(&self) -> Result<()> {
        info!("Creating table if it doesn't exist...");
        sqlx::query(
            r#"
            CREATE TABLE IF NOT EXISTS documents (
                name VARCHAR(128) NOT NULL,
                description VARCHAR(2000),
                embedding BLOB NOT NULL,
                VECTOR INDEX (embedding)
            )
            "#,
            )
            .execute(&self.pool)
            .await?;

        info!("Table structure verified");
        Ok(())
    }

    pub async fn process_file(&self, path: &Path) -> Result<()> {
        let content = self.read_file_content(path)?;
        info!("Successfully extracted text from file");

        let embedding = self.generate_embedding(&content)?;

        let doc = Document {
            name: path.file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string(),
                description: content,
                embedding,
        };

        self.save_document(&doc).await?;
        info!("Finished processing file: {}", path.display());
        Ok(())
    }

    fn read_file_content(&self, path: &Path) -> Result<String> {
        let extension = path.extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("");

        match extension.to_lowercase().as_str() {
            "pdf" => {
                info!("Processing PDF file");
                Ok(extract_text(path)?)
            },
            "txt" => {
                info!("Processing text file");
                Ok(std::fs::read_to_string(path)?)
            },
            _ => Err(anyhow::anyhow!("Unsupported file type: {}", extension)),
        }
    }

    fn generate_embedding(&self, text: &str) -> Result<Vec<f32>> {
        // Basic text vectorization using a simple word frequency approach
        let words: Vec<&str> = text.split_whitespace().collect();
        let mut word_count = std::collections::HashMap::new();

        for word in words {
            *word_count.entry(word.to_lowercase()).or_insert(0) += 1;
        }

        // Create a simple vector where each index corresponds to a unique word
        let unique_words: Vec<String> = word_count.keys().cloned().collect();
        let mut embedding = vec![0.0; unique_words.len()];

        for (i, word) in unique_words.iter().enumerate() {
            if let Some(&count) = word_count.get(word) {
                embedding[i] = count as f32; // Using word frequency as the embedding value
            }
        }

        Ok(embedding)
    }

    async fn save_document(&self, doc: &Document) -> Result<()> {
        sqlx::query(
            r#"
            INSERT INTO documents (name, description, embedding)
            VALUES (?, ?, VEC_FromText(?))
            "#,
            )
            .bind(&doc.name)
            .bind(&doc.description)
            .bind(self.format_vector(&doc.embedding))
            .execute(&self.pool)
            .await?;

        info!("Saved document: {}", doc.name);
        Ok(())
    }

    fn format_vector(&self, vector: &[f32]) -> String {
        format!("[{}]", vector.iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(","))
    }

    pub async fn search_similar_chunks(&self, query: &str, limit: i32) -> Result<Vec<SearchResult>> {
        info!("Generating embedding for query: {}", query);
        let query_embedding = self.generate_embedding(query)?;
        let query_vector = self.format_vector(&query_embedding);

        let results = sqlx::query(
            r#"
    SELECT
        name,
        description,
        1 - VEC_DISTANCE(embedding, VEC_FromText(?)) as similarity
    FROM documents
    ORDER BY similarity DESC
    LIMIT ?
    "#,
    )
            .bind(query_vector)
            .bind(limit)
            .map(|row: sqlx::mysql::MySqlRow| {
                SearchResult {
                    name: row.get("name"),
                    description: row.get("description"),
                    similarity: row.get::<Option<f32>, _>("similarity").unwrap_or(0.0), // Handle NULL values
                }
            })
        .fetch_all(&self.pool)
            .await?;

        Ok(results)
    }
}

fn print_search_results(results: &[SearchResult]) {
    println!("\nSearch Results:");
    println!("------------------------");

    for (i, result) in results.iter().enumerate() {
        println!("\n{}. Document: {}", i + 1, result.name);
        println!("Excerpt: {}", truncate_text(&result.description, 200));
        println!("------------------------");
    }
}

fn truncate_text(text: &str, max_length: usize) -> String {
    if text.len() <= max_length {
        text.to_string()
    } else {
        let mut truncated = text[..max_length].to_string();
        truncated.push_str("...");
        truncated
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenv::dotenv().ok();
    tracing_subscriber::fmt::init();

    let args = Args::parse();
    let db_url = std::env::var("DATABASE_URL")
        .unwrap_or_else(|_| "mysql://root:testvector@localhost:3306/vectordb".to_string());

    let app = App::new(&db_url).await?;

    match args.command {
        Commands::Index { file } => {
            app.process_file(&file).await?;
        }
        Commands::Search { query, limit } => {
            let results = app.search_similar_chunks(&query, limit).await?;
            print_search_results(&results);
        }
    }

    Ok(())
}

