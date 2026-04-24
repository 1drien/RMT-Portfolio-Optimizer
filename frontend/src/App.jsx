import React, { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, 
         BarChart, Bar, Legend, CartesianGrid, ReferenceLine, 
         Scatter, ComposedChart } from 'recharts';
import { TrendingDown, ShieldCheck } from 'lucide-react';

const TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "JPM", "V", "KO"];

// ✅ Composant dehors de App()
const MPChart = ({ data }) => {
  if (!data) return null;

  const curveData = data.curve.map(p => ({ x: p.x, curve: p.y }));
  const eigenData = data.eigenvalues.map(v => ({ x: v, eigen: 0 }));

  return (
    <div className="bg-white border border-[#D4CFC4] p-8 shadow-lg">
      <h3 className="text-xs font-bold uppercase text-[#0A192F] mb-2 tracking-[0.2em]">
        Marchenko-Pastur : Séparation Signal / Bruit
      </h3>

      <div className="flex gap-8 mb-6">
        <div>
          <p className="text-[10px] text-[#A39E93] uppercase tracking-widest">Facteur Q</p>
          <p className="text-xl font-light text-[#0A192F]">{data.Q}</p>
        </div>
        <div>
          <p className="text-[10px] text-[#A39E93] uppercase tracking-widest">λ max</p>
          <p className="text-xl font-light text-[#0A192F]">{data.lambda_max}</p>
        </div>
        <div>
          <p className="text-[10px] text-[#A39E93] uppercase tracking-widest">Valeurs propres signal</p>
          <p className="text-xl font-light text-[#C5A059]">{data.n_signal} / {data.eigenvalues.length}</p>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={300}>
        <ComposedChart margin={{ top: 10, right: 30, bottom: 20, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#F4F1EA" vertical={false} />
          <XAxis
            dataKey="x"
            type="number"
            domain={['auto', 'auto']}
            stroke="#A39E93"
            fontSize={10}
            tickLine={false}
            axisLine={false}
            tickFormatter={v => v.toFixed(2)}
            label={{ value: 'Valeur propre λ', position: 'insideBottom', offset: -10, fontSize: 10, fill: '#A39E93' }}
          />
          <YAxis
            stroke="#A39E93"
            fontSize={10}
            axisLine={false}
            tickLine={false}
            tickFormatter={v => v.toFixed(2)}
          />
          <Tooltip
            contentStyle={{ backgroundColor: '#FFFFFF', border: '1px solid #D4CFC4' }}
            formatter={(val) => val.toFixed(4)}
          />
          <Line
            data={curveData}
            type="monotone"
            dataKey="curve"
            stroke="#C0392B"
            strokeWidth={2}
            dot={false}
            name="Densité MP théorique"
          />
          <Scatter
            data={eigenData}
            dataKey="eigen"
            fill="#0A192F"
            name="Valeurs propres empiriques"
          />
          <ReferenceLine
            x={data.lambda_max}
            stroke="#C5A059"
            strokeWidth={2}
            strokeDasharray="5 5"
            label={{ value: `λmax = ${data.lambda_max}`, position: 'top', fontSize: 10, fill: '#C5A059' }}
          />
        </ComposedChart>
      </ResponsiveContainer>

      <div className="flex gap-6 mt-2 justify-center">
        <div className="flex items-center gap-2">
          <div className="w-4 h-0.5 bg-red-600"></div>
          <span className="text-[10px] text-[#A39E93]">Distribution MP théorique (bruit)</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-[#0A192F]"></div>
          <span className="text-[10px] text-[#A39E93]">Valeurs propres empiriques</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-4 h-0.5 bg-[#C5A059]"></div>
          <span className="text-[10px] text-[#A39E93]">Seuil λmax</span>
        </div>
      </div>
    </div>
  );
};

function App() {
  const [selected, setSelected] = useState(["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]);
  const [startDate, setStartDate] = useState("2018-01-01");
  const [endDate, setEndDate] = useState("2023-12-31");
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState(null);

  const startAnalysis = async () => {
    setLoading(true);
    try {
      const res = await fetch('http://localhost:8000/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ tickers: selected, start_date: startDate, end_date: endDate, split_ratio: 0.7 })
      });
      const result = await res.json();
      if (res.ok) setData(result);
      else alert("Erreur serveur");
    } catch (e) { alert("Impossible de joindre le serveur"); }
    setLoading(false);
  };

  return (
    <div className="flex h-screen bg-[#F4F1EA] text-[#1A1A1A] font-serif">
      <aside className="w-64 bg-[#0A192F] text-white p-8 flex flex-col border-r-4 border-[#C5A059]">
        <div className="mb-12 text-center">
          <ShieldCheck className="mx-auto mb-2 text-[#C5A059]" size={32} />
          <h1 className="text-[#C5A059] font-serif text-xl font-light tracking-[0.2em] uppercase">CY Asset Management</h1>
        </div>

        <div className="flex-1 space-y-8">
          <div>
            <p className="text-[10px] text-[#C5A059] font-bold uppercase mb-4 tracking-widest border-b border-[#C5A059]/30 pb-1">Period Selection</p>
            <input type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} className="w-full bg-[#112240] border border-[#C5A059]/20 p-2 rounded-sm text-xs text-white outline-none focus:border-[#C5A059]" />
            <input type="date" value={endDate} onChange={(e) => setEndDate(e.target.value)} className="w-full bg-[#112240] border border-[#C5A059]/20 p-2 rounded-sm text-xs mt-2 text-white outline-none focus:border-[#C5A059]" />
          </div>

          <div>
            <p className="text-[10px] text-[#C5A059] font-bold uppercase mb-4 tracking-widest border-b border-[#C5A059]/30 pb-1">Asset Universe</p>
            <div className="grid grid-cols-2 gap-2">
              {TICKERS.map(t => (
                <button key={t} onClick={() => setSelected(prev => prev.includes(t) ? prev.filter(x => x !== t) : [...prev, t])}
                  className={`p-2 text-[10px] font-bold border transition-all ${selected.includes(t) ? 'bg-[#C5A059] border-[#C5A059] text-[#0A192F]' : 'border-slate-700 text-slate-400 hover:border-[#C5A059]'}`}>{t}</button>
              ))}
            </div>
          </div>
        </div>

        <button onClick={startAnalysis} disabled={loading} className="w-full py-4 bg-[#C5A059] text-[#0A192F] font-black uppercase tracking-widest text-xs hover:bg-[#D4B475] transition-colors shadow-lg">
          {loading ? "ANALYZING..." : "GENERATE ALPHA"}
        </button>
      </aside>

      <main className="flex-1 p-10 overflow-y-auto space-y-10">
        {!data ? (
          <div className="h-full flex flex-col items-center justify-center opacity-30 text-[#0A192F]">
            <TrendingDown size={100} strokeWidth={1} />
            <p className="mt-4 font-serif italic tracking-widest">Awaiting Institutional Directives</p>
          </div>
        ) : (
          <>
            {/* KPI ROW */}
            <div className="grid grid-cols-6 gap-4">
              {[
                { label: "Benchmark Vol", val: data.metrics.vol_naive + "%", color: "text-slate-500" },
                { label: "Optimal RMT Vol", val: data.metrics.vol_rmt + "%", color: "text-[#0A192F]" },
                { label: "Stability Index", val: "+" + data.metrics.gain + "%", color: "text-[#C5A059]" },
                { label: "Sharpe Naive", val: data.metrics.sharpe_naive, color: "text-slate-500" },
                { label: "Sharpe RMT", val: data.metrics.sharpe_rmt, color: "text-[#C5A059]" },
                { label: "Risk VaR (95%)", val: data.metrics.var_95 + "%", color: "text-red-800" }
              ].map((kpi, i) => (
                <div key={i} className="bg-white border border-[#D4CFC4] p-5 shadow-md relative overflow-hidden">
                  <div className="absolute top-0 left-0 w-1 h-full bg-[#C5A059]"></div>
                  <p className="text-[9px] text-[#A39E93] font-bold uppercase tracking-widest mb-2">{kpi.label}</p>
                  <p className={`text-2xl font-light tracking-tighter ${kpi.color}`}>{kpi.val}</p>
                </div>
              ))}
            </div>

            {/* PERFORMANCE CHART */}
            <div className="bg-white border border-[#D4CFC4] p-8 shadow-lg">
              <h3 className="text-xs font-bold uppercase text-[#0A192F] mb-6 tracking-[0.2em] border-b border-[#F4F1EA] pb-4">Performance Analysis : RMT Strategy vs Market</h3>
              <ResponsiveContainer width="100%" height={350}>
                <LineChart data={data.chart_data}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#F4F1EA" vertical={false} />
                  <XAxis dataKey="date" hide />
                  <YAxis domain={['auto', 'auto']} stroke="#A39E93" fontSize={10} axisLine={false} tickLine={false} />
                  <Tooltip contentStyle={{ backgroundColor: '#FFFFFF', border: '1px solid #D4CFC4', borderRadius: '0px' }} />
                  <Legend verticalAlign="top" align="right" iconType="circle" />
                  <Line name="RMT Portfolio" type="monotone" dataKey="rmt" stroke="#C5A059" strokeWidth={3} dot={false} />
                  <Line name="S&P 500" type="monotone" dataKey="spy" stroke="#0A192F" strokeWidth={1} dot={false} strokeDasharray="5 5" />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* ✅ MARCHENKO-PASTUR */}
            <MPChart data={data.marchenko_pastur} />

            {/* HEATMAP & WEIGHTS */}
            <div className="grid grid-cols-2 gap-10 pb-10">
              <div className="bg-white border border-[#D4CFC4] p-8 shadow-lg">
                <h3 className="text-xs font-bold uppercase text-[#0A192F] mb-6 tracking-[0.2em]">Risk Correlation Matrix</h3>
                <div className="grid" style={{ gridTemplateColumns: `repeat(${data.heatmap.labels.length}, 1fr)` }}>
                  {data.heatmap.data.flat().map((val, i) => (
                    <div key={i} className="aspect-square border border-[#F4F1EA] flex items-center justify-center text-[9px] font-bold transition-all hover:bg-[#C5A059] hover:text-white"
                      style={{ backgroundColor: `rgba(197, 160, 89, ${Math.abs(val)})`, color: Math.abs(val) > 0.5 ? 'white' : '#0A192F' }}>{val}</div>
                  ))}
                </div>
              </div>

              <div className="bg-white border border-[#D4CFC4] p-8 shadow-lg h-80">
                <h3 className="text-xs font-bold uppercase text-[#0A192F] mb-6 tracking-[0.2em]">Asset Allocation Strategy</h3>
                <ResponsiveContainer width="100%" height="90%">
                  <BarChart data={data.weights}>
                    <XAxis dataKey="name" stroke="#A39E93" fontSize={10} axisLine={false} tickLine={false} />
                    <YAxis stroke="#A39E93" fontSize={10} axisLine={false} tickLine={false} />
                    <Tooltip cursor={{ fill: '#F4F1EA' }} contentStyle={{ backgroundColor: '#FFFFFF', border: '1px solid #D4CFC4' }} />
                    <Bar name="Weight" dataKey="rmt" fill="#0A192F" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </>
        )}
      </main>
    </div>
  );
}

export default App;