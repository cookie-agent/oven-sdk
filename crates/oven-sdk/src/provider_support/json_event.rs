//! Allocation-light decoding of hot streaming event payloads.
//!
//! Adapters classify most events through `JsonValue` lookups. Building that DOM
//! for every text delta dominates per-event cost, so these helpers let an
//! adapter decode the few hot event shapes into borrowed typed structs first
//! and fall back to its `JsonValue` path whenever the typed decode declines.
//! Callers keep all edge-case semantics (mistyped fields, duplicate keys) in
//! that fallback, so the typed fast path only has to be exact for the shapes it
//! accepts.

use std::{borrow::Cow, fmt, marker::PhantomData};

use serde::{
    Deserialize, Deserializer,
    de::{IgnoredAny, MapAccess, SeqAccess, Visitor, value::MapAccessDeserializer},
};

/// Decodes `T` only from a JSON object.
///
/// Derived structs also accept JSON arrays as positional fields, which a
/// `JsonValue` object lookup never would. Wrapping a nested struct field in
/// this type keeps typed decoding aligned with object lookups.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct JsonObject<T>(pub T);

impl<'de, T> Deserialize<'de> for JsonObject<T>
where
    T: Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct ObjectVisitor<T>(PhantomData<T>);

        impl<'de, T> Visitor<'de> for ObjectVisitor<T>
        where
            T: Deserialize<'de>,
        {
            type Value = JsonObject<T>;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a JSON object")
            }

            fn visit_map<A>(self, map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                T::deserialize(MapAccessDeserializer::new(map)).map(JsonObject)
            }
        }

        deserializer.deserialize_map(ObjectVisitor(PhantomData))
    }
}

/// Decodes a complete JSON object payload into `T`, or `None` when it does not fit.
///
/// `None` covers invalid JSON, non-object payloads, and any field that does not
/// match `T`; callers then take their `JsonValue` path. Ignored fields are only
/// syntax-checked, so a payload whose ignored fields hold numbers outside `f64`
/// range decodes here although a `JsonValue` parse would reject it.
#[must_use]
pub fn parse_json_object<'de, T>(data: &'de str) -> Option<T>
where
    T: Deserialize<'de>,
{
    serde_json::from_str::<JsonObject<T>>(data)
        .ok()
        .map(|object| object.0)
}

/// Records whether an object key was present, whatever its value (including `null`).
///
/// Use with `#[serde(default)]` to mirror `value.get(key).is_some()`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct KeyPresent(pub bool);

impl<'de> Deserialize<'de> for KeyPresent {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        IgnoredAny::deserialize(deserializer).map(|_| Self(true))
    }
}

/// Returns the string `type` field of a JSON object payload without building a DOM.
///
/// Mirrors `serde_json::from_str::<JsonValue>(data).ok()?.get("type")?.as_str()`:
/// invalid JSON, non-object payloads, a missing key, and a non-string value all
/// yield `None`, and the last of duplicate keys wins.
#[must_use]
pub fn json_type_tag(data: &str) -> Option<Cow<'_, str>> {
    serde_json::from_str::<TypeTag<'_>>(data).ok()?.0
}

struct TypeTag<'a>(Option<Cow<'a, str>>);

impl<'de> Deserialize<'de> for TypeTag<'de> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct TagVisitor;

        impl<'de> Visitor<'de> for TagVisitor {
            type Value = TypeTag<'de>;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("any JSON value")
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut tag = None;
                while let Some(key) = map.next_key::<Cow<'de, str>>()? {
                    if key == "type" {
                        tag = map.next_value::<MaybeStr<'de>>()?.0;
                    } else {
                        map.next_value::<IgnoredAny>()?;
                    }
                }
                Ok(TypeTag(tag))
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                while seq.next_element::<IgnoredAny>()?.is_some() {}
                Ok(TypeTag(None))
            }

            fn visit_bool<E>(self, _: bool) -> Result<Self::Value, E> {
                Ok(TypeTag(None))
            }

            fn visit_i64<E>(self, _: i64) -> Result<Self::Value, E> {
                Ok(TypeTag(None))
            }

            fn visit_u64<E>(self, _: u64) -> Result<Self::Value, E> {
                Ok(TypeTag(None))
            }

            fn visit_f64<E>(self, _: f64) -> Result<Self::Value, E> {
                Ok(TypeTag(None))
            }

            fn visit_str<E>(self, _: &str) -> Result<Self::Value, E> {
                Ok(TypeTag(None))
            }

            fn visit_unit<E>(self) -> Result<Self::Value, E> {
                Ok(TypeTag(None))
            }
        }

        deserializer.deserialize_any(TagVisitor)
    }
}

/// A JSON value that is kept only when it is a string.
struct MaybeStr<'a>(Option<Cow<'a, str>>);

impl<'de> Deserialize<'de> for MaybeStr<'de> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct StrVisitor;

        impl<'de> Visitor<'de> for StrVisitor {
            type Value = MaybeStr<'de>;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("any JSON value")
            }

            fn visit_borrowed_str<E>(self, value: &'de str) -> Result<Self::Value, E> {
                Ok(MaybeStr(Some(Cow::Borrowed(value))))
            }

            fn visit_str<E>(self, value: &str) -> Result<Self::Value, E> {
                Ok(MaybeStr(Some(Cow::Owned(value.to_owned()))))
            }

            fn visit_string<E>(self, value: String) -> Result<Self::Value, E> {
                Ok(MaybeStr(Some(Cow::Owned(value))))
            }

            fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                while map.next_entry::<IgnoredAny, IgnoredAny>()?.is_some() {}
                Ok(MaybeStr(None))
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                while seq.next_element::<IgnoredAny>()?.is_some() {}
                Ok(MaybeStr(None))
            }

            fn visit_bool<E>(self, _: bool) -> Result<Self::Value, E> {
                Ok(MaybeStr(None))
            }

            fn visit_i64<E>(self, _: i64) -> Result<Self::Value, E> {
                Ok(MaybeStr(None))
            }

            fn visit_u64<E>(self, _: u64) -> Result<Self::Value, E> {
                Ok(MaybeStr(None))
            }

            fn visit_f64<E>(self, _: f64) -> Result<Self::Value, E> {
                Ok(MaybeStr(None))
            }

            fn visit_unit<E>(self) -> Result<Self::Value, E> {
                Ok(MaybeStr(None))
            }
        }

        deserializer.deserialize_any(StrVisitor)
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use serde::Deserialize;

    use super::{JsonObject, KeyPresent, json_type_tag, parse_json_object};

    #[derive(Debug, Deserialize, PartialEq)]
    struct Event<'a> {
        #[serde(rename = "type", borrow)]
        kind: Cow<'a, str>,
        index: u64,
        #[serde(borrow)]
        delta: JsonObject<Delta<'a>>,
        #[serde(default)]
        error: KeyPresent,
    }

    #[derive(Debug, Deserialize, PartialEq)]
    struct Delta<'a> {
        #[serde(default, borrow)]
        text: Option<Cow<'a, str>>,
    }

    #[test]
    fn typed_object_borrows_unescaped_strings_and_owns_escaped_ones() {
        let event: Event<'_> = parse_json_object(
            r#"{"type":"content_block_delta","index":3,"delta":{"text":"a\"b"},"extra":[1,{"x":null}]}"#,
        )
        .unwrap();
        assert!(matches!(event.kind, Cow::Borrowed("content_block_delta")));
        assert_eq!(event.index, 3);
        assert!(matches!(event.delta.0.text, Some(Cow::Owned(ref text)) if text == "a\"b"));
        assert_eq!(event.error, KeyPresent(false));
    }

    #[test]
    fn arrays_and_mistyped_fields_decline_the_typed_decode() {
        assert!(parse_json_object::<Event<'_>>(r#"["content_block_delta",0,{}]"#).is_none());
        assert!(
            parse_json_object::<Event<'_>>(r#"{"type":"t","index":0,"delta":["text"]}"#).is_none()
        );
        assert!(
            parse_json_object::<Event<'_>>(r#"{"type":"t","index":0,"delta":{"text":1}}"#)
                .is_none()
        );
        assert!(parse_json_object::<Event<'_>>(r#"{"type":"t","index":-1,"delta":{}}"#).is_none());
        assert!(parse_json_object::<Event<'_>>(r#"{"type":"t","index":0,"delta":{}} x"#).is_none());
    }

    #[test]
    fn key_presence_includes_null_values() {
        let event: Event<'_> =
            parse_json_object(r#"{"type":"t","index":0,"delta":{},"error":null}"#).unwrap();
        assert_eq!(event.error, KeyPresent(true));
    }

    #[test]
    fn type_tag_matches_json_value_lookup() {
        for (data, expected) in [
            (r#"{"type":"ping"}"#, Some("ping")),
            (
                r#"{"a":{"type":"inner"},"type":"outer","b":[1,2]}"#,
                Some("outer"),
            ),
            (r#"{"type":"first","type":"last"}"#, Some("last")),
            (r#"{"type":"first","type":7}"#, None),
            (r#"{"type":"tick"}"#, Some("tick")),
            (r#"{"type":null}"#, None),
            (r#"{"kind":"ping"}"#, None),
            (r#"["type","ping"]"#, None),
            (r#""ping""#, None),
            ("not json", None),
            (r#"{"type":"ping"} trailing"#, None),
        ] {
            let via_value = serde_json::from_str::<serde_json::Value>(data)
                .ok()
                .and_then(|value| value.get("type")?.as_str().map(str::to_owned));
            assert_eq!(via_value.as_deref(), expected, "{data}");
            assert_eq!(json_type_tag(data).as_deref(), expected, "{data}");
        }
    }
}
