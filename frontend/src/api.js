async function getJson(path, signal) {
  let response;
  try {
    response = await fetch(path, { signal });
  } catch (error) {
    if (signal?.aborted) {
      throw new DOMException("Request aborted", "AbortError");
    }
    throw new Error(`API request failed. Is the FastAPI server running on http://127.0.0.1:8000? (${error.message})`);
  }

  const text = await response.text();
  let payload = {};
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      throw new Error(`API returned a non-JSON response (${response.status}).`);
    }
  }

  if (!response.ok) {
    if (payload.detail) {
      throw new Error(payload.detail);
    }
    if (!text && response.status >= 500) {
      throw new Error(`API returned ${response.status} with no error body. Check that FastAPI is running on http://127.0.0.1:8000 and inspect the backend terminal logs.`);
    }
    throw new Error(`Request failed (${response.status})`);
  }
  return payload;
}

export const fetchBootstrap = (signal) => getJson("/api/bootstrap", signal);

export const fetchDashboard = ({ ticker, start, end }, signal) => {
  const params = new URLSearchParams({ ticker, start, end });
  return getJson(`/api/dashboard?${params}`, signal);
};
