import { useEffect, useMemo, useRef, useState } from "react";
import * as echarts from "echarts";
import { fetchBootstrap, fetchDashboard } from "./api";

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

function CardHead({ eyebrow, title, note }) {
  return (
    <div className="card-head">
      <div>
        <span>{eyebrow}</span>
        <h3>{title}</h3>
        {note && <p className="card-note">{note}</p>}
      </div>
    </div>
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
    if (!keyword) return tickers.slice(0, 12);
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
        <div className="readout-line"><span>Watch next</span><p>{readout.watch}</p></div>
      </div>
      {readout.alignment ? <Alignment alignment={readout.alignment} /> : (
        <div className="readout-facts">
          <div><span>Return</span><strong>{readout.facts.return}</strong></div>
          <div><span>Risk tone</span><strong>{readout.facts.risk_tone}</strong></div>
          <div><span>8-K impact</span><strong>{readout.facts.eight_k_impact}</strong></div>
          <div><span>Alignment</span><strong>Needs prior 10-K</strong></div>
        </div>
      )}
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
        <div><span>10-K excess return</span><strong>{formatPct(alignment.excess_return)}</strong></div>
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

function Charts({ charts }) {
  const prices = charts?.prices || [];
  const events = charts?.events || [];
  const eventSeries = (type) => events
    .filter((event) => event.type === type && event.chart_date && event.close != null)
    .map((event) => ({ value: [event.chart_date, event.close], filingDate: event.date, details: event.details }));
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
        name: "10-K filing",
        type: "scatter",
        symbol: "diamond",
        symbolSize: 15,
        itemStyle: { color: "#000", borderColor: "#8FFE01", borderWidth: 2 },
        data: eventSeries("10-K"),
        markLine: { symbol: "none", silent: true, lineStyle: { color: "rgba(5,5,5,.28)", type: "dashed", width: 1 }, data: events.filter((event) => event.type === "10-K" && event.chart_date).map((event) => ({ xAxis: event.chart_date })) },
        tooltip: { formatter: (params) => `<strong>10-K filing</strong><br/>Filing date: ${params.data.filingDate}<br/>${params.data.details || ""}<br/>Chart date: ${params.value[0]}` }
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
      <section className="card chart-card"><CardHead eyebrow="Trend" title="Open vs Close Prices" note="Use wheel or trackpad to zoom. Drag inside the plot to move the time window." />{prices.length ? <EChart option={priceOption} /> : <Empty>No price data in range.</Empty>}</section>
      <section className="card chart-card"><CardHead eyebrow="Liquidity" title="Trading Volume" />{prices.length ? <EChart option={volumeOption} /> : <Empty>No volume data in range.</Empty>}</section>
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
              <h3>{section.current_date} vs {section.previous_date}</h3>
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
  if (!reactions.length) return <Empty>No 10-K filings found for this ticker and date range.</Empty>;
  return (
    <div className="reaction-list">
      {reactions.map((reaction) => (
        <article className="reaction-item" key={reaction.date}>
          <div className="reaction-head">
            <div><span className="event-type">10-K</span><strong>{reaction.date}</strong><p className="reaction-subtitle">Sections: {reaction.sections}</p></div>
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

function FilingSections({ sections, kpis }) {
  const mdnaSentiment = kpiValue(kpis, "MD&A Sentiment", sections.mdna.sentiment);
  const riskSentiment = kpiValue(kpis, "Risk Sentiment", sections.risk.sentiment);

  return (
    <>
      <div className="sentiment-tag">Sentiment - MD&amp;A: {mdnaSentiment} | Risk: {riskSentiment}</div>
      <div className="grid ai-summary-grid">
        <SummaryPanel title="MD&A Summary" summary={sections.mdna.summary} />
        <SummaryPanel title="Risk Summary" summary={sections.risk.summary} />
      </div>
      <div className="grid info filing-section-grid">
        <TextPanel eyebrow="Filing stream" title="MD&A - Management Discussion" section={sections.mdna} />
        <TextPanel eyebrow="Risk factors" title="Risk Sections" section={sections.risk} />
      </div>
    </>
  );
}

function TextPanel({ eyebrow, title, section }) {
  return (
    <section className="card info-card">
      <CardHead eyebrow={eyebrow} title={title} />
      <div className="filing-actions">
        {section.date && <p className="filing-date">Filing date: {section.date}</p>}
        {section.url && <a className="event-link" href={section.url} rel="noreferrer" target="_blank">SEC filing</a>}
      </div>
      <div className="filing-text">{section.text}</div>
    </section>
  );
}

function SummaryPanel({ title, summary }) {
  return (
    <section className="card summary-card">
      <div className="ai-change-summary">
        <div className="ai-change-head"><span>AI summary</span><strong>{title}</strong></div>
        <div className="ai-change-body">{summary || "No summary available."}</div>
      </div>
    </section>
  );
}

export default function App() {
  const [bootstrap, setBootstrap] = useState(null);
  const [selection, setSelection] = useState({ ticker: "", start: "", end: "" });
  const [dashboard, setDashboard] = useState(null);
  const [bootError, setBootError] = useState("");
  const [dashboardError, setDashboardError] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const controller = new AbortController();
    fetchBootstrap(controller.signal).then((data) => {
      setBootstrap(data);
      setSelection({ ticker: data.default_ticker || "", start: data.start_date || "", end: data.end_date || "" });
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

  if (bootError) return <main className="page-shell"><Empty>{bootError}</Empty></main>;
  return (
    <main className="page-shell">
      <header className="topbar" data-parallax-depth="4"><div className="brand"><div className="brand-mark">F</div><span>FinSight</span></div><div className="topbar-action">React + FastAPI</div></header>
      <section className="hero-card">
        <div className="hero-meta" data-parallax-depth="10"><span className="eyebrow">Market pulse</span><h1>Narrative vs. Market</h1><p>Analyzing how 10-K filing language aligns with stock price movement, market reaction, and risk signals.</p></div>
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
        <section className="card readout-card"><CardHead eyebrow="Final insight" title="Market Readout" note="Price action, filing tone, 10-K reaction, and 8-K event impact in one view." /><MarketReadout readout={dashboard.market_readout} /></section>
        {dashboard.price_coverage?.warning && <Empty>{dashboard.price_coverage.warning}</Empty>}
        <Charts charts={dashboard.charts} />
        <section className="card comparison-card"><CardHead eyebrow="Year over year" title="Filing Comparison" note="Selected filing versus the previous available 10-K language." /><Comparison sections={dashboard.comparison} /></section>
        <section className="card events-card"><CardHead eyebrow="Event detail" title="8-K Events" /><EightKEvents events={dashboard.eight_k_events} /></section>
        <section className="card reaction-card"><CardHead eyebrow="Market reaction" title="Price Reaction After 10-K" note="Returns anchor on the next trading day and compare with the S&P 500." /><Reactions reactions={dashboard.reactions} /></section>
        <FilingSections sections={dashboard.sections} kpis={dashboard.kpis || []} />
      </>}
    </main>
  );
}
