# Provider error bodies

All provider HTTP error classifiers use `provider_support::sanitize_error_body`.
The same classifiers handle provider SSE errors and Bedrock EventStream exception
payloads and error headers. This covers OpenAI (including compatible Chat and
Responses), Anthropic and its protocol variants, Google, Vertex, Bedrock, Azure
Chat and Responses, Cohere, and Open Responses.

`ModelError.message` and its `Display` remain stable, generic descriptions.
Consumers obtain the provider's reason from `diagnostics.sanitized_body.text()`.
`Debug` continues to hide the body. Serde exposes the **already scrubbed** text in
the existing shape:

```json
{
  "sanitized_body": {
    "text": "{\"error\":{\"message\":\"unsupported dimension; Bearer [REDACTED]\",\"details\":{\"expected\":1024,\"api_key\":\"[REDACTED]\"}}}",
    "truncated": false
  }
}
```

This example shows only the body field of the diagnostics object. There is no new
error-envelope schema or change to classification, retryability, Retry-After,
request ID, vendor code, status, or stage selection.

## Retention and scrubbing

- Valid JSON passed to the classifier retains its envelope, unknown fields, arrays, numeric values, and
  useful provider messages. Object order and whitespace may change. Credential
  values (including credential objects/arrays) become the string `[REDACTED]`.
  Existing stream handlers may first extract the error portion of an event; this
  change preserves that portion, not unrelated response/output content.
- Nested credential keys are matched case insensitively, ignoring punctuation.
  These include API keys, authorization/proxy authorization, cookies, passwords,
  secrets, credentials, session/auth fields, token suffixes, private keys, AWS
  access keys, and signed URL credential/signature parameters. Header objects
  using `name`/`value` or `key`/`value` are also recognized, regardless of field
  order or case. This association stays within the same object in prefixed JSON,
  incomplete envelopes, and embedded/encoded JSON strings. Complete credential
  values or their incomplete retained prefixes are redacted; unrelated `value`
  fields, including sibling headers, remain available.
- String values and non-JSON gateway bodies retain scrubbed text. The scanner
  removes Bearer/Basic credential values, secret `key=value` / `key: value`
  assignments, query secrets (including percent-encoded key names), and common
  OpenAI/Anthropic, Google, AWS, GitHub, Slack, and JWT credential forms. Quoted
  values and truncated credential objects are handled. Complete escaped JSON
  strings are decoded even in otherwise malformed/truncated JSON.
- HTTP(S) URL authority userinfo is replaced with `[REDACTED]@`, removing both
  username and password while preserving the host, path, and harmless query
  parameters. This covers percent-encoded credential components and JSON-escaped
  URLs in JSON, plain text, and prefixed bodies, including every RFC 3986 userinfo
  subdelimiter (`!$&'()*+,;=`). Apostrophes within userinfo do not end the authority;
  whitespace, controls, path/query/fragment separators, and surrounding double
  quotes or HTML delimiters still stop the scan before unrelated prose `@` signs.
  Only diagnostic text is changed;
  request URLs and error classifications retain their existing behavior.
- `sanitize_error_body_with_secrets` additionally accepts credential values already
  available to its caller. It scrubs them in decoded JSON keys/string values and
  text, including complete escaped strings in prefixed bodies. For example, a
  supplied `review-secret` is removed from `"review\u002dsecret"`. Empty values are
  ignored and overlapping matches prefer the longest value. Existing adapters use
  the original helper without a supplied credential list; this additive entry
  point does not introduce credential discovery or propagation through streams.
- Unicode control characters and bidi embedding/isolate controls become spaces.
  Invalid UTF-8 becomes the Unicode replacement character. HTML stays text;
  consumers must escape it if rendering HTML.
- Empty input has no body. OpenAI previously produced an empty body container in
  this case. Open Responses now retains the original error envelope instead of
  flattening selected error fields, and Azure retains it instead of a synthetic
  `codes` array. Other adapters no longer reduce bodies to type/status/code alone
  or omit all provider text.

## Bounds and limitations

Diagnostic input is limited to the first **65,536 bytes before parsing**. Scrubbed
output is independently limited to **65,536 bytes at a UTF-8 boundary**. Truncation
is recorded when input or output exceeds the cap, or an HTTP body reader reports
more bytes than were retained. Stream byte counts are cumulative and do not by
themselves mark an error event's body as truncated. A truncated body may be invalid
JSON; consumers should display it as text. Redaction expansion and UTF-8 repair
can also make the output reach its cap.

Pass the provider body to the sanitizer before adding display prefixes or
truncation markers. Known-value matching runs on decoded text before control
normalization and output truncation; replacing raw JSON bytes alone is insufficient.
Quoted credential values are scanned past escaped quotes, and credential objects
or arrays are removed as a whole, including their retained prefix if incomplete.

Fallback header associations use span tables and explicit stacks bounded by the
retained input size, without reparsing every nested JSON suffix. Embedded JSON
string decoding is limited to eight recursive unescaping layers per text scan;
deeper quoted values are redacted. An unterminated JSON string can be decoded from
its retained prefix, dropping at most a trailing incomplete escape/surrogate pair
(12 bytes). These conservative redactions do not themselves mark a body truncated;
the byte-cap/transport rules above determine that flag.

Existing transport read policies remain: some readers drain and count the full
response, while Anthropic stops once its retained buffer fills. Anthropic now
reads up to **65,537 bytes** to distinguish exactly-at-cap input from overflow.
Its byte count remains the number observed before stopping, not the server's
complete response size. This diagnostic cap does not change stream frame limits,
classification parsing, or timeout/cancellation/read-failure behavior.

This is best-effort credential scrubbing, **not a public-data or PII guarantee**.
Unlabelled secrets with unfamiliar formats, arbitrary echoed request credentials,
encoded/obfuscated values, prompts, personal data, and proprietary provider details
can remain. Request authentication material is not passed to the shared error
classifier; this change does not add credential storage to streams or change
provider APIs to propagate it. There is no general discovery of arbitrary echoed
secrets. Supplied-secret matching is exact; it does not cover arbitrary encodings
or partial unlabelled values cut off by the input cap. Identifiers outside
`sanitized_body` retain their existing validation and
are outside this body's scrubbing contract.

Structural recognition requires the identifying syntax in the retained input:
for example, a header name occurring entirely beyond the cap cannot identify an
earlier `value`, and URL userinfo whose `@` lies beyond the cap may be unrecognized.
Fully encoded URL schemes and arbitrary malformed/obfuscated syntax are not
generally decoded. The sanitizer does not redact every field named `value`.

`SanitizedBody::new` and deserialization remain bounded containers, not redactors
for arbitrary application input. The SDK adapters sanitize **before** constructing
their bodies, so serialization does not rely on a downstream application to scrub
provider body credentials. Consumers should still control diagnostic access and
retention. Protocol/read failures without a classified provider error retain their
existing generic diagnostics.

## Verification and consumer integration

The shared conformance matrix runs in every adapter for 400/403/429, JSON and
HTML/text bodies, response and stream stages, credential removal, useful unknown
fields, UTF-8 bounds, and truncation. Core tests exercise nested credentials,
escaped and malformed JSON, headers, query parameters, control characters,
serialization, Debug, and transport-versus-stream byte counts. Regression tests
also cover supplied secrets escaped in JSON, escaped quotes in prefixed password
fields, prefixed/truncated nested credential values, and sanitization before
presentation and output truncation. Stream tests cover
the actual provider error paths, including Bedrock exception payloads/error headers
and both OpenAI/Azure protocols. Tests use local fixtures; live model tests remain
ignored.

The shared adapter matrix additionally checks header pairs in prefixed, truncated,
embedded and double-encoded JSON, and HTTP(S) userinfo in JSON/plain/prefixed text
and escaped URLs. Core regression names for this review are
`header_pairs_survive_prefixed_truncated_and_encoded_envelopes`,
`incomplete_header_pair_values_are_redacted_through_the_input_cap`,
`header_pair_associations_stay_within_their_object`,
`http_url_userinfo_is_redacted_in_json_text_and_encoded_strings`, and
`fragment_scanning_and_string_decoding_have_bounded_depth`.

Consumers already reading `sanitized_body` gain useful provider explanations when
they adopt a revision containing this change. Consumers must tolerate JSON or plain
text and should not depend on the former synthetic body shapes. No dependency pin,
application manifest, release, commit, or publication is part of this change.
