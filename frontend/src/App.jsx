import { useEffect, useMemo, useRef, useState } from "react";
import * as echarts from "echarts";
import { fetchBootstrap, fetchDashboard, subscribeFilingAlert } from "./api";

const presets = [
  ["YTD", "ytd"],
  ["1Y", "1y"],
  ["3Y", "3y"],
  ["5Y", "5y"],
  ["All", "all"]
];

function addYears(iso, years) {
  const value = new Date(`${iso}T12:00:00`);
  value.setFullYear(value.getFullYear() - years);
  return value.toISOString().slice(0, 10);
}

function presetRange(preset, bounds) {
  if (preset === "all") return { start: bounds.start_date, end: bounds.end_date };
  if (preset === "ytd") return { start: `${bounds.end_date.slice(0, 4)}-01-01`, end: bounds.end_date };
  return { start: addYears(bounds.end_date, Number(preset[0])), end: bounds.end_date };
}

function formatPct(value) {
  return value == null ? "N/A" : `${value >= 0 ? "+" : ""}${(value * 100).toFixed(2)}%`;
}

function CardHead({ eyebrow, title, note, action }) {
  return (
    <div className="card-head">
      <div>
        <span>{eyebrow}</span>
        <h3>{title}</h3>
        {note && <p className="card-note">{note}</p>}
      </div>
      {action}
    </div>
  );
}

function CollapsibleSection({ id, className = "", eyebrow, title, note, collapsed, onToggle, children }) {
  return (
    <section className={`card ${className} ${collapsed ? "is-collapsed" : ""}`}>
      <CardHead
        eyebrow={eyebrow}
        title={title}
        note={note}
        action={(
          <button
            aria-expanded={!collapsed}
            className="collapse-toggle"
            onClick={() => onToggle(id)}
            type="button"
          >
            {collapsed ? "Show" : "Hide"}
          </button>
        )}
      />
      {!collapsed && children}
    </section>
  );
}

function Loading({ label = "Loading current selection" }) {
  return (
    <div className="react-loading">
      <i />
      <span>{label}</span>
    </div>
  );
}

function Empty({ children }) {
  return <div className="empty-state">{children}</div>;
}

function TickerSearch({ tickers, value, onChange }) {
  const [query, setQuery] = useState(value);
  const [open, setOpen] = useState(false);
  const inputRef = useRef(null);

  useEffect(() => {
    setQuery(value);
  }, [value]);

  const matches = useMemo(() => {
    const keyword = query.trim().toUpperCase();
    if (!keyword) return tickers;
    return tickers.filter((ticker) => ticker.toUpperCase().includes(keyword)).slice(0, 12);
  }, [query, tickers]);

  const selectTicker = (ticker) => {
    setQuery(ticker);
    setOpen(false);
    onChange(ticker);
  };

  const handleChange = (event) => {
    const nextQuery = event.target.value.toUpperCase();
    setQuery(nextQuery);
    setOpen(true);
    if (tickers.includes(nextQuery)) onChange(nextQuery);
  };

  const handleKeyDown = (event) => {
    if (event.key === "Enter" && matches.length) {
      event.preventDefault();
      selectTicker(matches[0]);
    }
    if (event.key === "Escape") {
      setOpen(false);
      inputRef.current?.blur();
    }
  };

  return (
    <div className="ticker-search">
      <input
        aria-label="Search ticker"
        autoComplete="off"
        onBlur={() => window.setTimeout(() => setOpen(false), 120)}
        onChange={handleChange}
        onFocus={() => setOpen(true)}
        onKeyDown={handleKeyDown}
        placeholder="Search ticker"
        ref={inputRef}
        value={query}
      />
      {open && (
        <div className="ticker-menu" role="listbox">
          {matches.length ? matches.map((ticker) => (
            <button key={ticker} onMouseDown={() => selectTicker(ticker)} type="button">
              {ticker}
            </button>
          )) : <div className="ticker-empty">No ticker found</div>}
        </div>
      )}
    </div>
  );
}

function Kpis({ kpis, loading }) {
  return (
    <div className="hero-highlight">
      <div className="hero-kpis kpi-shell">
        {loading ? <Loading label="Refreshing market pulse" /> : (
          <div className="kpi-grid">
            {kpis.map((item) => (
              <article className={`kpi-item ${item.tone}`} key={item.label}>
                <span>{item.label}</span>
                <strong>{item.value}</strong>
                <em>{item.detail}</em>
              </article>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function MarketReadout({ readout }) {
  if (!readout) return <Empty>No readout available.</Empty>;
  return (
    <div className="readout-grid">
      <div className={`readout-stance ${readout.stance_tone}`}>
        <span>Overall stance</span>
        <strong>{readout.stance}</strong>
        <em>Score {readout.score > 0 ? "+" : ""}{readout.score}</em>
      </div>
      <div className="readout-copy">
        <div className="readout-line"><span>Why it matters</span><p>{readout.why}</p></div>
        <div className="readout-line"><span>Key driver</span><p>{readout.driver}</p></div>
        {readout.disclosure && <div className="readout-line"><span>Latest disclosure</span><p>{readout.disclosure.label}</p></div>}
        <div className="readout-line"><span>Watch next</span><p>{readout.watch}</p></div>
      </div>
      {readout.alignment ? <Alignment alignment={readout.alignment} /> : (
        <div className="readout-facts">
          <div><span>Return</span><strong>{readout.facts.return}</strong></div>
          <div><span>Risk tone</span><strong>{readout.facts.risk_tone}</strong></div>
          <div><span>8-K impact</span><strong>{readout.facts.eight_k_impact}</strong></div>
          <div><span>Revenue</span><strong>{readout.facts.revenue || "N/A"}</strong></div>
          <div><span>Gross margin</span><strong>{readout.facts.gross_margin || "N/A"}</strong></div>
          <div><span>Cash flow</span><strong>{readout.facts.cash_flow || "N/A"}</strong></div>
          <div><span>Disclosure age</span><strong>{readout.facts.disclosure || "N/A"}</strong></div>
        </div>
      )}
    </div>
  );
}

function FinancialStatements({ financials }) {
  if (!financials || financials.status !== "ready") {
    return <Empty>{financials?.note || "No financial statement highlights available."}</Empty>;
  }
  return (
    <div className="financial-grid">
      <div className="financial-source">
        <span>{financials.source.form_type}</span>
        <strong>{financials.source.date}</strong>
        <a className="event-link secondary" href={financials.source.url} rel="noreferrer" target="_blank">Source filing</a>
      </div>
      <div className="financial-metrics">
        {financials.metrics.map((metric) => (
          <div className={`financial-metric ${metric.tone}`} key={metric.label}>
            <span>{metric.label}</span>
            <strong>{metric.value}</strong>
            <p>{metric.detail}</p>
            {metric.delta != null && <em>{formatPct(metric.delta)}</em>}
          </div>
        ))}
      </div>
    </div>
  );
}

function ValuationAnchor({ anchor }) {
  if (!anchor) return <Empty>No valuation anchor available.</Empty>;
  return (
    <div className="anchor-grid">
      <div className={`anchor-status ${anchor.tone}`}>
        <span>{anchor.status}</span>
        <strong>{anchor.fair_value ? `$${anchor.fair_value.toFixed(2)}` : "N/A"}</strong>
        <p>{anchor.summary}</p>
        <em>{anchor.latest_price ? `Latest close $${anchor.latest_price.toFixed(2)}` : "No price anchor"}</em>
      </div>
      <div className="anchor-tiers">
        {(anchor.tiers || []).map((tier) => (
          <div className="anchor-tier" key={tier.label}>
            <span>{tier.label}</span>
            <strong>{tier.price != null ? `$${tier.price.toFixed(2)}` : `$${tier.price_low.toFixed(2)} - $${tier.price_high.toFixed(2)}`}</strong>
            <p>{tier.detail}</p>
            <em>{tier.discount != null ? `${formatPct(tier.discount)} vs latest` : "N/A"}</em>
          </div>
        ))}
      </div>
      <div className="anchor-facts">
        <div><span>Safety Discount</span><strong>{anchor.margin_of_safety != null ? formatPct(anchor.margin_of_safety) : "N/A"}</strong></div>
        <div><span>Range low/high</span><strong>{anchor.range_low ? `$${anchor.range_low.toFixed(2)} / $${anchor.range_high.toFixed(2)}` : "N/A"}</strong></div>
      </div>
      <div className="anchor-rationale">
        {anchor.rationale.map((item) => <p key={item}>{item}</p>)}
        <em>{anchor.disclaimer}</em>
      </div>
    </div>
  );
}

function Alignment({ alignment }) {
  return (
    <div className="alignment-panel">
      <div className={`alignment-head ${alignment.tone}`}>
        <span>Narrative vs Market Alignment</span>
        <strong>{alignment.output}</strong>
        <p>{alignment.why}</p>
      </div>
      <div className="alignment-factors">
        <div><span>Filing tone change</span><strong>{alignment.tone_change}</strong></div>
        <div><span>New risk language</span><strong>{alignment.new_risk_count}</strong></div>
        <div><span>Disclosure excess return</span><strong>{formatPct(alignment.excess_return)}</strong></div>
        <div><span>Sentiment shift</span><strong>{alignment.sentiment_shift ?? "N/A"}</strong></div>
        <div><span>Market reaction</span><strong>{alignment.reaction}</strong></div>
      </div>
    </div>
  );
}

function EChart({ option }) {
  const host = useRef(null);
  useEffect(() => {
    const node = host.current;
    if (!node) return undefined;
    const chart = echarts.init(node, null, { renderer: "canvas" });
    chart.setOption(option, true);
    const resize = () => chart.resize();
    window.addEventListener("resize", resize);
    return () => {
      window.removeEventListener("resize", resize);
      chart.dispose();
    };
  }, [option]);
  return <div className="react-echart" ref={host} />;
}

function Charts({ charts, collapsed, onToggle }) {
  const prices = charts?.prices || [];
  const events = charts?.events || [];
  const eventSeries = (type) => events
    .filter((event) => event.type === type && event.chart_date && event.close != null)
    .map((event) => ({ value: [event.chart_date, event.close], filingDate: event.date, details: event.details }));
  const disclosureEvents = events
    .filter((event) => ["10-Q", "10-K"].includes(event.type) && event.chart_date && event.close != null)
    .map((event) => ({ value: [event.chart_date, event.close], filingDate: event.date, details: event.details, type: event.type }));
  const chartBase = {
    backgroundColor: "transparent",
    animationDuration: 450,
    textStyle: { fontFamily: "Urbanist, system-ui, sans-serif", color: "#111" },
    grid: { left: 56, right: 24, top: 42, bottom: 72 },
    legend: { top: 4, left: 0, itemGap: 18 },
    toolbox: { right: 0, top: 0, feature: { dataZoom: { yAxisIndex: "none" }, restore: {} } },
    dataZoom: [
      { type: "inside", xAxisIndex: 0, filterMode: "none", zoomOnMouseWheel: true, moveOnMouseMove: true },
      { type: "slider", xAxisIndex: 0, height: 28, bottom: 18, borderColor: "rgba(5,5,5,.08)", fillerColor: "rgba(114,1,255,.12)", handleStyle: { color: "#7201FF" } }
    ],
    xAxis: { type: "time", axisLine: { lineStyle: { color: "rgba(17,17,17,.16)" } }, axisTick: { show: false }, splitLine: { show: false } },
    yAxis: { type: "value", scale: true, axisLine: { show: false }, axisTick: { show: false }, splitLine: { lineStyle: { color: "rgba(17,17,17,.08)" } } }
  };
  const priceOption = {
    ...chartBase,
    tooltip: { trigger: "axis", axisPointer: { type: "cross" }, valueFormatter: (value) => typeof value === "number" ? `$${value.toFixed(2)}` : value },
    yAxis: { ...chartBase.yAxis, axisLabel: { formatter: (value) => `$${Number(value).toFixed(0)}` } },
    series: [
      { name: "Open", type: "line", smooth: true, showSymbol: false, data: prices.map((row) => [row.date, row.open]), lineStyle: { width: 3, color: "#8FFE01" } },
      { name: "Close", type: "line", smooth: true, showSymbol: false, data: prices.map((row) => [row.date, row.close]), lineStyle: { width: 3, color: "#7201FF" } },
      {
        name: "10-Q / 10-K filing",
        type: "scatter",
        symbol: "diamond",
        symbolSize: 15,
        itemStyle: { color: "#000", borderColor: "#8FFE01", borderWidth: 2 },
        data: disclosureEvents,
        markLine: { symbol: "none", silent: true, lineStyle: { color: "rgba(5,5,5,.28)", type: "dashed", width: 1 }, data: events.filter((event) => ["10-Q", "10-K"].includes(event.type) && event.chart_date).map((event) => ({ xAxis: event.chart_date })) },
        tooltip: { formatter: (params) => `<strong>${params.data.type} filing</strong><br/>Filing date: ${params.data.filingDate}<br/>${params.data.details || ""}<br/>Chart date: ${params.value[0]}` }
      },
      {
        name: "8-K filing",
        type: "scatter",
        symbolSize: 10,
        itemStyle: { color: "#7201FF", borderColor: "#fff", borderWidth: 2 },
        data: eventSeries("8-K"),
        tooltip: { formatter: (params) => `<strong>8-K filing</strong><br/>Filing date: ${params.data.filingDate}<br/>${params.data.details || ""}<br/>Chart date: ${params.value[0]}` }
      }
    ]
  };
  const volumeOption = {
    ...chartBase,
    legend: { show: false },
    tooltip: { trigger: "axis", axisPointer: { type: "shadow" }, valueFormatter: (value) => Number(value).toLocaleString() },
    yAxis: { ...chartBase.yAxis, axisLabel: { formatter: (value) => `${(Number(value) / 1_000_000).toFixed(0)}M` } },
    series: [{ name: "Volume", type: "bar", data: prices.map((row) => [row.date, row.volume]), itemStyle: { color: "#7201FF", borderRadius: [4, 4, 0, 0] }, large: true }]
  };
  return (
    <div className="grid charts">
      <CollapsibleSection id="price-chart" className="chart-card" eyebrow="Trend" title="Open vs Close Prices" note="Use wheel or trackpad to zoom. Drag inside the plot to move the time window." collapsed={collapsed["price-chart"]} onToggle={onToggle}>
        {prices.length ? <EChart option={priceOption} /> : <Empty>No price data in range.</Empty>}
      </CollapsibleSection>
      <CollapsibleSection id="volume-chart" className="chart-card" eyebrow="Liquidity" title="Trading Volume" collapsed={collapsed["volume-chart"]} onToggle={onToggle}>
        {prices.length ? <EChart option={volumeOption} /> : <Empty>No volume data in range.</Empty>}
      </CollapsibleSection>
    </div>
  );
}

function Comparison({ sections }) {
  return (
    <div className="comparison-grid">
      {sections.map((section) => section.status !== "ready" ? <Empty key={section.name}>{section.message}</Empty> : (
        <article className="comparison-section" key={section.name}>
          <div className="comparison-section-head">
            <div>
              <span className="event-type">{section.name}</span>
              <h3>{section.current_form_type || "Filing"} {section.current_date} vs {section.previous_form_type || "Filing"} {section.previous_date}</h3>
              <div className="comparison-links">
                <a className="event-link" href={section.current_url} rel="noreferrer" target="_blank">Current filing</a>
                <a className="event-link secondary" href={section.previous_url} rel="noreferrer" target="_blank">Previous filing</a>
              </div>
            </div>
          </div>
          <div className="comparison-metrics">
            <Metric label="Words" value={section.metrics.word_count.toLocaleString()} delta={section.metrics.word_delta} />
            <Metric label="Readability" value={section.metrics.readability ?? "N/A"} delta={section.metrics.readability_delta} />
            <Metric label="Narrative tone" value={section.metrics.tone} delta={section.metrics.sentiment_delta} />
          </div>
          <div className="ai-change-summary">
            <div className="ai-change-head"><span>AI what changed</span><strong>{section.name}</strong></div>
            <div className="ai-change-body">{section.ai_change_summary}</div>
          </div>
          <LanguageList title={`New ${section.name} language`} lines={section.added} />
          <LanguageList title={`Removed ${section.name} language`} lines={section.removed} />
        </article>
      ))}
    </div>
  );
}

function Metric({ label, value, delta }) {
  return <div className="comparison-metric"><span>{label}</span><strong>{value}</strong><em>{delta == null ? "" : `${delta > 0 ? "+" : ""}${delta}`}</em></div>;
}

function LanguageList({ title, lines }) {
  return (
    <div className="language-list">
      <h4>{title}</h4>
      {lines.length ? <ul>{lines.map((line) => <li key={line}>{line}</li>)}</ul> : <p className="muted-copy">No major sentence-level changes detected.</p>}
    </div>
  );
}

function EightKEvents({ events }) {
  if (!events.length) return <Empty>No 8-K events found for this ticker and date range.</Empty>;
  return (
    <div className="event-list">
      {events.map((event) => (
        <article className="event-item" key={`${event.date}-${event.url || event.items}`}>
          <div className="event-item-head">
            <div><span className="event-type">{event.form_type}</span><strong>{event.date}</strong></div>
            <span className={`impact-label ${event.impact_tone}`}>{event.impact}</span>
          </div>
          <p className="event-summary">{event.summary}</p>
          {event.horizons && (
            <>
              <div className={`reaction-signal ${event.reaction_tone || "neutral"}`}>{event.reaction_label || "Insufficient data"}</div>
              <div className="reaction-grid event-reaction-grid">
                {["1", "5", "30"].map((horizon) => <ReactionMetric key={horizon} label={`${horizon}D`} result={event.horizons[horizon]} />)}
              </div>
            </>
          )}
          <details className="event-details">
            <summary>Details</summary>
            <div className="event-items">{event.items}</div>
            <p className="event-preview">{event.preview}</p>
            <div className="event-source">Sources: {event.sources}</div>
            {event.url && <a className="event-link" href={event.url} rel="noreferrer" target="_blank">SEC filing</a>}
          </details>
        </article>
      ))}
    </div>
  );
}

function Reactions({ reactions }) {
  if (!reactions.length) return <Empty>No 10-Q or 10-K filings found for this ticker and date range.</Empty>;
  return (
    <div className="reaction-list">
      {reactions.map((reaction) => (
        <article className="reaction-item" key={reaction.date}>
          <div className="reaction-head">
            <div><span className="event-type">{reaction.form_type || "10-K"}</span><strong>{reaction.date}</strong><p className="reaction-subtitle">Sections: {reaction.sections}</p></div>
            <a className="event-link" href={reaction.url} rel="noreferrer" target="_blank">SEC filing</a>
          </div>
          <div className={`reaction-signal ${reaction.tone}`}>{reaction.label}</div>
          <div className="reaction-grid">
            {["1", "5", "30"].map((horizon) => <ReactionMetric key={horizon} label={`${horizon}D`} result={reaction.horizons[horizon]} />)}
          </div>
        </article>
      ))}
    </div>
  );
}

function ReactionMetric({ label, result }) {
  return <div className="reaction-metric"><span>{label}</span><strong>{formatPct(result.stock)}</strong><em>vs S&amp;P {formatPct(result.excess)}</em></div>;
}

function kpiValue(kpis, label, fallback) {
  return kpis.find((item) => item.label === label)?.value || fallback;
}

function FilingSections({ sections, kpis, collapsed, onToggle }) {
  const mdnaSentiment = kpiValue(kpis, "MD&A Tone", sections.mdna.sentiment);
  const riskSentiment = kpiValue(kpis, "Risk Disclosure", sections.risk.sentiment);

  return (
    <>
      <div className="sentiment-tag">Sentiment - MD&amp;A: {mdnaSentiment} | Risk: {riskSentiment}</div>
      <div className="grid ai-summary-grid">
        <SummaryPanel id="mdna-summary" title="MD&A Summary" summary={sections.mdna.summary} collapsed={collapsed["mdna-summary"]} onToggle={onToggle} />
        <SummaryPanel id="risk-summary" title="Risk Summary" summary={sections.risk.summary} collapsed={collapsed["risk-summary"]} onToggle={onToggle} />
      </div>
      <div className="grid info filing-section-grid">
        <TextPanel id="mdna-text" eyebrow="Filing stream" title="MD&A - Management Discussion" section={sections.mdna} collapsed={collapsed["mdna-text"]} onToggle={onToggle} />
        <TextPanel id="risk-text" eyebrow="Risk factors" title="Risk Sections" section={sections.risk} collapsed={collapsed["risk-text"]} onToggle={onToggle} />
      </div>
    </>
  );
}

function TextPanel({ id, eyebrow, title, section, collapsed, onToggle }) {
  return (
    <CollapsibleSection id={id} className="info-card" eyebrow={eyebrow} title={title} collapsed={collapsed} onToggle={onToggle}>
      <div className="filing-actions">
        {section.date && <p className="filing-date">Filing date: {section.date}</p>}
        {section.url && <a className="event-link" href={section.url} rel="noreferrer" target="_blank">SEC filing</a>}
      </div>
      <div className="filing-text">{section.text}</div>
    </CollapsibleSection>
  );
}

function SummaryPanel({ id, title, summary, collapsed, onToggle }) {
  return (
    <CollapsibleSection id={id} className="summary-card" eyebrow="AI summary" title={title} collapsed={collapsed} onToggle={onToggle}>
      <div className="ai-change-summary">
        <div className="ai-change-body">{summary || "No summary available."}</div>
      </div>
    </CollapsibleSection>
  );
}

function AlertSignup({ ticker }) {
  const [email, setEmail] = useState("");
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");
  const [saving, setSaving] = useState(false);
  const [hidden, setHidden] = useState(false);
  const railRef = useRef(null);
  const storageKey = ticker ? `finsight-alert-hidden:${ticker}` : "finsight-alert-hidden";

  useEffect(() => {
    setStatus("");
    setError("");
    setHidden(window.localStorage.getItem(storageKey) === "1");
  }, [storageKey]);

  useEffect(() => {
    let frame = 0;
    const followScroll = () => {
      window.cancelAnimationFrame(frame);
      frame = window.requestAnimationFrame(() => {
        if (!railRef.current) return;
        const y = window.scrollY + window.innerHeight * 0.52;
        railRef.current.style.setProperty("--alert-y", `${y}px`);
      });
    };
    followScroll();
    window.addEventListener("scroll", followScroll, { passive: true });
    window.addEventListener("resize", followScroll);
    return () => {
      window.cancelAnimationFrame(frame);
      window.removeEventListener("scroll", followScroll);
      window.removeEventListener("resize", followScroll);
    };
  }, []);

  const hideAlert = () => {
    window.localStorage.setItem(storageKey, "1");
    setHidden(true);
  };

  const submit = (event) => {
    event.preventDefault();
    setSaving(true);
    setError("");
    setStatus("");
    subscribeFilingAlert({ email, ticker }).then(() => {
      setStatus(`Subscribed ${email} to ${ticker} filing alerts.`);
      setEmail("");
      hideAlert();
    }).catch((err) => {
      setError(err.message);
    }).finally(() => setSaving(false));
  };

  if (hidden) return null;

  return (
    <div className="alert-float" ref={railRef}>
      <button className="alert-orb" type="button" aria-label="Filing alert signup">!</button>
      <section className="alert-popover">
        <button className="alert-close" type="button" aria-label="Hide filing alert signup" onClick={hideAlert}>x</button>
        <CardHead eyebrow="Email alerts" title="Filing Alerts" note="Get an email when a new 8-K, 10-Q, or 10-K is detected for the selected ticker." />
        <form className="alert-form" onSubmit={submit}>
          <label className="field">Email<input autoComplete="email" placeholder="you@example.com" type="email" value={email} onChange={(event) => setEmail(event.target.value)} required /></label>
          <button disabled={saving || !ticker} type="submit">{saving ? "Saving..." : `Alert me for ${ticker || "ticker"}`}</button>
        </form>
        {status && <p className="alert-status">{status}</p>}
        {error && <p className="alert-error">{error}</p>}
      </section>
    </div>
  );
}

export default function App() {
  const [bootstrap, setBootstrap] = useState(null);
  const [selection, setSelection] = useState({ ticker: "", start: "", end: "" });
  const [dashboard, setDashboard] = useState(null);
  const [bootError, setBootError] = useState("");
  const [dashboardError, setDashboardError] = useState("");
  const [loading, setLoading] = useState(true);
  const [collapsedSections, setCollapsedSections] = useState({});

  useEffect(() => {
    const controller = new AbortController();
    fetchBootstrap(controller.signal).then((data) => {
      setBootstrap(data);
      setSelection({ ticker: data.default_ticker || "", start: data.default_start_date || data.start_date || "", end: data.end_date || "" });
    }).catch((error) => {
      if (error.name !== "AbortError") setBootError(error.message);
    });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (!selection.ticker || !selection.start || !selection.end) return undefined;
    const controller = new AbortController();
    setLoading(true);
    setDashboard(null);
    setDashboardError("");
    fetchDashboard(selection, controller.signal).then(setDashboard).catch((error) => {
      if (error.name !== "AbortError") setDashboardError(error.message);
    }).finally(() => setLoading(false));
    return () => controller.abort();
  }, [selection]);

  const invalidRange = selection.start && selection.end && selection.start > selection.end;
  const applyPreset = (preset) => setSelection((value) => ({ ...value, ...presetRange(preset, bootstrap) }));
  const toggleSection = (id) => {
    setCollapsedSections((value) => ({ ...value, [id]: !value[id] }));
  };

  if (bootError) return <main className="page-shell"><Empty>{bootError}</Empty></main>;
  return (
    <main className="page-shell">
      <header className="topbar" data-parallax-depth="4"><div className="brand"><div className="brand-mark">F</div><span>FinSight</span></div><div className="topbar-action">React + FastAPI</div></header>
      <section className="hero-card">
        <div className="hero-meta" data-parallax-depth="10"><span className="eyebrow">Market pulse</span><h1>Narrative vs. Market</h1><p>Analyzing how recent 10-Q/10-K language aligns with stock price movement, event reactions, and risk signals.</p></div>
        <div data-parallax-depth="18">
          <Kpis kpis={dashboard?.kpis || []} loading={loading || !dashboard} />
        </div>
      </section>
      <section className="controls-card card">
        <label className="field">Select Ticker<TickerSearch tickers={bootstrap?.tickers || []} value={selection.ticker} onChange={(ticker) => setSelection((value) => ({ ...value, ticker }))} /></label>
        <div className="field date-field"><label>Select Date Range</label><div className="native-range"><input type="date" min={bootstrap?.start_date} max={bootstrap?.end_date} value={selection.start} onChange={(event) => setSelection((value) => ({ ...value, start: event.target.value }))} /><input type="date" min={bootstrap?.start_date} max={bootstrap?.end_date} value={selection.end} onChange={(event) => setSelection((value) => ({ ...value, end: event.target.value }))} /></div></div>
        <div className="field range-preset-field"><label>Quick Ranges</label><div className="range-presets">{presets.map(([label, value]) => <button key={value} onClick={() => applyPreset(value)} type="button">{label}</button>)}</div></div>
      </section>
      {invalidRange && <Empty>Start date must be on or before end date.</Empty>}
      {dashboardError && <Empty>{dashboardError}</Empty>}
      {loading && !dashboard && <section className="card loading-card"><Loading /></section>}
      {dashboard && <>
        <CollapsibleSection id="market-readout" className="readout-card" eyebrow="Evidence-weighted" title="Market Readout" note="Price action, latest disclosure freshness, short-window reaction, and 8-K event impact in one view." collapsed={collapsedSections["market-readout"]} onToggle={toggleSection}>
          <MarketReadout readout={dashboard.market_readout} />
        </CollapsibleSection>
        <CollapsibleSection id="valuation-anchor" className="anchor-card" eyebrow="Research anchor" title="Valuation Anchor" note="Fair value, watch price, and buy zone derived from price range, market readout, filing risk, and financial statement signals." collapsed={collapsedSections["valuation-anchor"]} onToggle={toggleSection}>
          <ValuationAnchor anchor={dashboard.valuation_anchor} />
        </CollapsibleSection>
        <CollapsibleSection id="financial-statements" className="financial-card" eyebrow="Statement pulse" title="Financial Statements" note="Revenue, margin, cash flow, and debt/liquidity signals extracted from the latest periodic filing." collapsed={collapsedSections["financial-statements"]} onToggle={toggleSection}>
          <FinancialStatements financials={dashboard.financials} />
        </CollapsibleSection>
        {dashboard.price_coverage?.warning && <Empty>{dashboard.price_coverage.warning}</Empty>}
        <Charts charts={dashboard.charts} collapsed={collapsedSections} onToggle={toggleSection} />
        <CollapsibleSection id="filing-comparison" className="comparison-card" eyebrow="Filing history" title="Filing Comparison" note="Selected periodic filing versus the previous available filing language." collapsed={collapsedSections["filing-comparison"]} onToggle={toggleSection}>
          <Comparison sections={dashboard.comparison} />
        </CollapsibleSection>
        <CollapsibleSection id="eight-k-events" className="events-card" eyebrow="Event detail" title="8-K Events" collapsed={collapsedSections["eight-k-events"]} onToggle={toggleSection}>
          <EightKEvents events={dashboard.eight_k_events} />
        </CollapsibleSection>
        <CollapsibleSection id="price-reaction" className="reaction-card" eyebrow="Market reaction" title="Price Reaction After Disclosure" note="Returns anchor on the next trading day after 10-Q or 10-K filings and compare with the S&P 500." collapsed={collapsedSections["price-reaction"]} onToggle={toggleSection}>
          <Reactions reactions={dashboard.reactions} />
        </CollapsibleSection>
        <FilingSections sections={dashboard.sections} kpis={dashboard.kpis || []} collapsed={collapsedSections} onToggle={toggleSection} />
      </>}
      <AlertSignup ticker={selection.ticker} />
    </main>
  );
}
