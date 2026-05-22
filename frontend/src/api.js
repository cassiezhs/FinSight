async function getJson(path, signal) {
  const response = await fetch(path, { signal });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.detail || `Request failed (${response.status})`);
  }
  return payload;
}

export const fetchBootstrap = (signal) => getJson("/api/bootstrap", signal);

export const fetchDashboard = ({ ticker, start, end }, signal) => {
  const params = new URLSearchParams({ ticker, start, end });
  return getJson(`/api/dashboard?${params}`, signal);
};
