//! Locally enforceable, surface-specific Azure media constraints.

use oven_sdk::{ErrorStage, FilePart, FileSource, HistoryTurn, InputPart, ModelError, Request};

const MAX_IMAGE_BYTES: usize = 20 * 1024 * 1024;
const CHAT_MAX_IMAGES: usize = 10;
const RESPONSES_MAX_IMAGES: usize = 50;
const RESPONSES_COMBINED_BYTES: usize = 50 * 1024 * 1024;
const RESPONSES_PDF_BYTES: usize = 50 * 1024 * 1024;

#[derive(Clone, Copy)]
pub(crate) enum Surface {
    Chat,
    Responses,
}

pub(crate) fn is_image(media_type: &str) -> bool {
    media_type.starts_with("image/")
}

pub(crate) fn supported_image(media_type: &str) -> bool {
    matches!(
        media_type,
        "image/png" | "image/jpeg" | "image/webp" | "image/gif"
    )
}

pub(crate) fn validate_request_media(
    request: &Request,
    surface: Surface,
) -> Result<(), ModelError> {
    let mut image_count = 0_usize;
    let mut image_bytes = 0_usize;
    let mut pdf_bytes = 0_usize;
    for turn in &request.history {
        let HistoryTurn::User(message) = turn else {
            continue;
        };
        for part in &message.content {
            let InputPart::File(file) = part else {
                continue;
            };
            if is_image(&file.media_type) {
                image_count = image_count
                    .checked_add(1)
                    .ok_or_else(|| media_error("Azure image count overflow"))?;
                image_bytes = image_bytes
                    .checked_add(validate_image(file)?)
                    .ok_or_else(|| media_error("Azure aggregate image size overflow"))?;
            } else if file.media_type == "application/pdf" {
                if matches!(surface, Surface::Chat) {
                    return Err(media_error(
                        "Azure Chat input supports images but not PDF files",
                    ));
                }
                pdf_bytes = pdf_bytes
                    .checked_add(validate_pdf(file)?)
                    .ok_or_else(|| media_error("Azure aggregate PDF size overflow"))?;
            }
        }
    }
    let max_images = match surface {
        Surface::Chat => CHAT_MAX_IMAGES,
        Surface::Responses => RESPONSES_MAX_IMAGES,
    };
    if image_count > max_images {
        return Err(media_error(match surface {
            Surface::Chat => "Azure Chat supports at most 10 input images",
            Surface::Responses => "Azure Responses supports at most 50 input images",
        }));
    }
    if matches!(surface, Surface::Responses) && image_bytes >= RESPONSES_COMBINED_BYTES {
        return Err(media_error(
            "Azure Responses inline images must total less than 50 MiB",
        ));
    }
    if pdf_bytes >= RESPONSES_COMBINED_BYTES {
        return Err(media_error(
            "Azure Responses inline PDFs must total less than 50 MiB",
        ));
    }
    Ok(())
}

fn validate_image(file: &FilePart) -> Result<usize, ModelError> {
    if !supported_image(&file.media_type) {
        return Err(media_error(
            "Azure image input supports only PNG, JPEG, WebP, and non-animated GIF",
        ));
    }
    match &file.source {
        FileSource::Bytes(bytes) => {
            if bytes.len() > MAX_IMAGE_BYTES {
                return Err(media_error("Azure input image exceeds 20 MiB"));
            }
            if file.media_type == "image/gif" && gif_frame_count(bytes) != Some(1) {
                return Err(media_error(
                    "Azure GIF input must be a valid non-animated GIF",
                ));
            }
            Ok(bytes.len())
        }
        FileSource::Url(_) if file.media_type == "image/gif" => Err(media_error(
            "Azure GIF URLs are unsupported because animation cannot be validated locally",
        )),
        FileSource::Url(_) => Ok(0),
        FileSource::Text(_) | FileSource::ProviderReference { .. } => {
            Err(media_error("unsupported Azure image source"))
        }
    }
}

fn validate_pdf(file: &FilePart) -> Result<usize, ModelError> {
    match &file.source {
        FileSource::Bytes(bytes) if bytes.len() < RESPONSES_PDF_BYTES => Ok(bytes.len()),
        FileSource::Bytes(_) => Err(media_error(
            "Azure Responses PDF files must be smaller than 50 MiB",
        )),
        FileSource::Url(_) => Ok(0),
        FileSource::Text(_) | FileSource::ProviderReference { .. } => Err(media_error(
            "Azure Responses PDF input requires bytes or a direct URL",
        )),
    }
}

fn gif_frame_count(bytes: &[u8]) -> Option<usize> {
    if bytes.len() < 13 || !matches!(&bytes[..6], b"GIF87a" | b"GIF89a") {
        return None;
    }
    let mut offset = 13;
    if bytes[10] & 0x80 != 0 {
        offset += 3 * (1usize << ((bytes[10] & 0x07) + 1));
    }
    let mut frames = 0;
    while offset < bytes.len() {
        match bytes[offset] {
            0x2c => {
                frames += 1;
                if frames > 1 || offset + 10 > bytes.len() {
                    return Some(frames);
                }
                let packed = bytes[offset + 9];
                offset += 10;
                if packed & 0x80 != 0 {
                    offset += 3 * (1usize << ((packed & 0x07) + 1));
                }
                offset = offset.checked_add(1)?;
                offset = skip_sub_blocks(bytes, offset)?;
            }
            0x21 => {
                offset = offset.checked_add(2)?;
                offset = skip_sub_blocks(bytes, offset)?;
            }
            0x3b => return Some(frames),
            _ => return None,
        }
    }
    None
}

fn skip_sub_blocks(bytes: &[u8], mut offset: usize) -> Option<usize> {
    loop {
        let length = *bytes.get(offset)? as usize;
        offset = offset.checked_add(1)?;
        if length == 0 {
            return Some(offset);
        }
        offset = offset.checked_add(length)?;
        if offset > bytes.len() {
            return None;
        }
    }
}

fn media_error(message: &str) -> ModelError {
    ModelError::unsupported(message).with_stage(ErrorStage::RequestEncoding)
}
