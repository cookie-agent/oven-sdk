//! Azure OpenAI route and identity constants.

use std::fmt;

use oven_sdk::{ModelError, ModelErrorKind};
use serde::{Deserialize, Deserializer, Serialize, Serializer, de::Error as _};

pub(crate) mod chat;
pub(crate) mod responses;

/// Stable Azure OpenAI provider identity.
pub const AZURE_OPENAI_PROVIDER_ID: &str = "azure.openai";
/// Stable Azure OpenAI Chat replay/adapter identity.
pub const AZURE_OPENAI_CHAT_ADAPTER_ID: &str = "oven.azure.openai.chat";
/// Stable Azure OpenAI Responses replay/adapter identity.
pub const AZURE_OPENAI_RESPONSES_ADAPTER_ID: &str = "oven.azure.openai.responses";

/// A validated dated Azure OpenAI data-plane API version.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct AzureApiVersion {
    label: String,
    year: u16,
    month: u8,
    day: u8,
}

impl AzureApiVersion {
    /// Validates a `YYYY-MM-DD` or `YYYY-MM-DD-preview` API version.
    pub fn new(value: impl Into<String>) -> Result<Self, ModelError> {
        let value = value.into();
        let date = match value.strip_suffix("-preview") {
            Some(date) => date,
            None => &*value,
        };
        let bytes = date.as_bytes();
        let valid = bytes.len() == 10
            && bytes[4] == b'-'
            && bytes[7] == b'-'
            && bytes
                .iter()
                .enumerate()
                .all(|(index, byte)| index == 4 || index == 7 || byte.is_ascii_digit());
        if !valid {
            return Err(ModelError::new(
                ModelErrorKind::InvalidRequest,
                "Azure API version must be YYYY-MM-DD or YYYY-MM-DD-preview",
            ));
        }
        let year = date[..4]
            .parse::<u16>()
            .map_err(|_| invalid_api_version())?;
        let month = date[5..7]
            .parse::<u8>()
            .map_err(|_| invalid_api_version())?;
        let day = date[8..10]
            .parse::<u8>()
            .map_err(|_| invalid_api_version())?;
        if year == 0 || month == 0 || month > 12 || day == 0 || day > days_in_month(year, month) {
            return Err(invalid_api_version());
        }
        Ok(Self {
            label: value,
            year,
            month,
            day,
        })
    }

    /// Returns the exact API-version label.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.label
    }

    pub(crate) fn supports_responses(&self) -> bool {
        (self.year, self.month, self.day) >= (2025, 3, 1)
    }
}

impl Serialize for AzureApiVersion {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&self.label)
    }
}

impl<'de> Deserialize<'de> for AzureApiVersion {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        Self::new(value).map_err(D::Error::custom)
    }
}

impl fmt::Display for AzureApiVersion {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.label)
    }
}

/// Typed Azure OpenAI inference route family.
#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "version")]
pub enum AzureApiRoute {
    /// Current GA `/openai/v1` route with no `api-version` query.
    #[default]
    V1,
    /// Current `/openai/v1` preview route using `api-version=preview`.
    V1Preview,
    /// Legacy deployment-based route with an exact dated API version.
    Dated(AzureApiVersion),
}

impl AzureApiRoute {
    pub(crate) fn endpoint(
        &self,
        origin: &url::Url,
        deployment: &str,
        endpoint: Endpoint,
    ) -> Result<url::Url, ModelError> {
        let mut url = origin.clone();
        let base = origin.path().trim_end_matches('/');
        let suffix = match (self, endpoint) {
            (Self::V1 | Self::V1Preview, Endpoint::Chat) => "/openai/v1/chat/completions",
            (Self::V1 | Self::V1Preview, Endpoint::Responses) => "/openai/v1/responses",
            (Self::Dated(_), Endpoint::Chat) => {
                let encoded = percent_encode_segment(deployment);
                url.set_path(&format!(
                    "{base}/openai/deployments/{encoded}/chat/completions"
                ));
                if let Self::Dated(version) = self {
                    url.query_pairs_mut()
                        .append_pair("api-version", version.as_str());
                }
                return Ok(url);
            }
            (Self::Dated(version), Endpoint::Responses) => {
                if !version.supports_responses() {
                    return Err(ModelError::unsupported(
                        "this dated Azure API version predates the Responses API",
                    ));
                }
                "/openai/responses"
            }
        };
        url.set_path(&format!("{base}{suffix}"));
        match self {
            Self::V1 => {}
            Self::V1Preview => {
                url.query_pairs_mut().append_pair("api-version", "preview");
            }
            Self::Dated(version) => {
                url.query_pairs_mut()
                    .append_pair("api-version", version.as_str());
            }
        }
        Ok(url)
    }
}

#[derive(Clone, Copy, Eq, PartialEq)]
pub(crate) enum Endpoint {
    Chat,
    Responses,
}

impl Endpoint {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Chat => "chat_completions",
            Self::Responses => "responses",
        }
    }
}

fn invalid_api_version() -> ModelError {
    ModelError::invalid_request(
        "Azure API version must be a real YYYY-MM-DD date with only an optional -preview suffix",
    )
}

const fn days_in_month(year: u16, month: u8) -> u8 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap_year(year) => 29,
        2 => 28,
        _ => 0,
    }
}

const fn is_leap_year(year: u16) -> bool {
    year.is_multiple_of(4) && (!year.is_multiple_of(100) || year.is_multiple_of(400))
}

fn percent_encode_segment(value: &str) -> String {
    url::form_urlencoded::byte_serialize(value.as_bytes())
        .collect::<String>()
        .replace('+', "%20")
}
