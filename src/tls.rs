use std::error::Error as StdError;
use std::io;
use std::net::ToSocketAddrs;
use std::path::PathBuf;
use std::sync::Arc;

use argh::FromArgs;
use rustls::pki_types::pem::PemObject;
use rustls::pki_types::{CertificateDer, ServerName};
use tokio::io::AsyncWriteExt;
use tokio::net::TcpStream;
use tokio_rustls::{rustls, TlsConnector};
use tokio_rustls::client::TlsStream;


pub async fn connect_tls(
    host: &str,
    port: u16,
    domain: Option<&str>,
    cafile: Option<&PathBuf>,
) -> Result<TlsStream<TcpStream>, Box<dyn StdError + Send + Sync + 'static>>

{
    let addr = (host, port).to_socket_addrs()?.next().ok_or_else(|| {
        io::Error::new(io::ErrorKind::NotFound, "Could not resolve address")
    })?;

    let domain = domain.unwrap_or(host);
    let content = format!("GET / HTTP/1.0\r\nHost: {}\r\n\r\n", domain);

    let mut root_cert_store = rustls::RootCertStore::empty();
    if let Some(cafile) = cafile {
        for cert in CertificateDer::pem_file_iter(cafile)? {
            root_cert_store.add(cert?)?;
        }
    } else {
        root_cert_store.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
    }

    let config = rustls::ClientConfig::builder()
        .with_root_certificates(root_cert_store)
        .with_no_client_auth();
    let connector = TlsConnector::from(Arc::new(config));

    let stream = TcpStream::connect(&addr).await?;


    let domain = ServerName::try_from(domain)?.to_owned();
    let mut stream = connector.connect(domain, stream).await?;
    stream.write_all(content.as_bytes()).await?;

    Ok(stream)
}


pub async fn example() -> Result<(), Box<dyn StdError + Send + Sync + 'static>> {
    let options: Options = argh::from_env();

    connect_tls(
        &options.host,
        options.port,
        options.domain.as_deref(),
        options.cafile.as_ref(),
    ).await?;

    Ok(())
}


/// Tokio Rustls client interface.
#[derive(FromArgs)]
struct Options {
    /// host
    #[argh(positional)]
    host: String,

    /// port
    #[argh(option, short = 'p', default = "443")]
    port: u16,

    /// domain
    #[argh(option, short = 'd')]
    domain: Option<String>,

    /// cafile
    #[argh(option, short = 'c')]
    cafile: Option<PathBuf>,
}