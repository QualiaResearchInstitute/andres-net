import React from "react";

export function Section({ title, children }: { title: React.ReactNode; children: React.ReactNode }) {
  return (
    <div className="col-span-2 rounded-2xl bg-zinc-900/60 border border-zinc-800 p-3 shadow-lg">
      <div className="text-sm font-semibold tracking-wide text-zinc-200 mb-2">{title}</div>
      <div className="grid grid-cols-2 gap-2">{children}</div>
    </div>
  );
}

export function Row({
  label,
  children,
  full = false,
}: {
  label: React.ReactNode;
  children: React.ReactNode;
  full?: boolean;
}) {
  return (
    <div className={`flex items-center gap-2 ${full ? "col-span-2" : "col-span-1"}`}>
      <div className="text-xs text-zinc-400 min-w-24">{label}</div>
      <div className="flex-1">{children}</div>
    </div>
  );
}

export function Range({
  value,
  min,
  max,
  step,
  onChange,
}: {
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
}) {
  return (
    <input
      type="range"
      className="w-full accent-cyan-400"
      min={min}
      max={max}
      step={step}
      value={value}
      onChange={(event) => onChange(parseFloat((event.target as HTMLInputElement).value))}
    />
  );
}

export function Toggle({
  value,
  onChange,
  label,
}: {
  value: boolean;
  onChange: (v: boolean) => void;
  label?: React.ReactNode;
}) {
  return (
    <label className="inline-flex items-center gap-2 select-none">
      {label && <span className="text-xs text-zinc-400">{label}</span>}
      <input type="checkbox" checked={value} onChange={(event) => onChange((event.target as HTMLInputElement).checked)} />
    </label>
  );
}
