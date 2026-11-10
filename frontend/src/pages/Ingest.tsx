import { useCallback, useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'

import { ApiError, api, formatDuration, type RawRun } from '../api'
import { ErrorNote } from '../components/ErrorNote'

const LAYOUTS = [
  { id: 'auto', label: 'Automático' },
  { id: 'face', label: 'Persona' },
  { id: 'split', label: 'Pantalla dividida' },
  { id: 'blur', label: 'Juego con blur' },
] as const

const PROVIDERS = [
  { id: 'mlx', label: 'Gemma en el M4' },
  { id: 'openai', label: 'OpenAI' },
  { id: 'ollama', label: 'Ollama' },
  { id: 'gemini', label: 'Gemini' },
  { id: 'anthropic', label: 'Claude' },
] as const

export function Ingest() {
  const navigate = useNavigate()
  const fileInput = useRef<HTMLInputElement>(null)
  const [url, setUrl] = useState('')
  const [file, setFile] = useState<File | null>(null)
  const [dragging, setDragging] = useState(false)
  const [lang, setLang] = useState('es')
  const [provider, setProvider] = useState('mlx')
  const [model, setModel] = useState('')
  const [layout, setLayout] = useState('auto')
  const [top, setTop] = useState(20)
  const [minS, setMinS] = useState(15)
  const [maxS, setMaxS] = useState(60)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<ApiError | Error | null>(null)
  const [runs, setRuns] = useState<RawRun[]>([])

  useEffect(() => {
    api.listRawRuns().then(setRuns).catch(() => undefined)
  }, [])

  const start = useCallback(async () => {
    if (!url.trim() && !file) return
    setBusy(true)
    setError(null)
    const form = new FormData()
    if (file) form.append('file', file)
    if (url.trim()) form.append('url', url.trim())
    form.append('lang', lang)
    form.append('provider', provider)
    form.append('model', model)
    form.append('layout', layout)
    form.append('top', String(top))
    form.append('min_s', String(minS))
    form.append('max_s', String(maxS))
    try {
      const run = await api.startRawRun(form)
      navigate(`/runs/${run.id}`)
    } catch (err) {
      setError(err as Error)
    } finally {
      setBusy(false)
    }
  }, [file, lang, layout, maxS, minS, model, navigate, provider, top, url])

  return (
    <div className="pt-14">
      <div className="rise max-w-3xl">
        <p className="eyebrow">Local · GPU del M4 · Premiere</p>
        <h1 className="mt-5 font-display text-[clamp(2.75rem,7vw,5.5rem)] leading-[0.95] text-ink-100">
          El video entra.
          <br />
          <span className="italic text-sodium-500">Los momentos</span> salen.
        </h1>
        <p className="mt-6 max-w-xl text-sm leading-relaxed text-ink-400">
          Transcribe, marca cambios de tema, pausas y picos, y Gemma (o OpenAI) se queda
          con los que valen. Cada clip sale en 9:16, listo para Premiere.
        </p>
      </div>

      <div className="mt-16 grid gap-x-16 gap-y-12 lg:grid-cols-[1.35fr_1fr]">
        <section className="rise" style={{ animationDelay: '90ms' }}>
          <label htmlFor="url" className="eyebrow">
            Enlace de YouTube
          </label>
          <input
            id="url"
            className="field mt-3 font-display text-xl md:text-2xl"
            placeholder="https://youtube.com/watch?v=…"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            autoComplete="off"
            spellCheck={false}
            disabled={busy}
          />

          <p className="eyebrow mt-10">O un archivo</p>
          <div
            onDragOver={(e) => {
              e.preventDefault()
              setDragging(true)
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault()
              setDragging(false)
              const dropped = e.dataTransfer.files[0]
              if (dropped) setFile(dropped)
            }}
            className={[
              'mt-3 flex items-end justify-between gap-4 border-b py-4',
              dragging ? 'border-sodium-500' : 'border-ink-700',
            ].join(' ')}
          >
            <div className="min-w-0">
              <p className="truncate text-sm text-ink-200">
                {file ? file.name : 'Suelta un mp4, mov o mkv'}
              </p>
              <p className="mt-1 text-xs text-ink-500">
                {file ? 'Se procesa donde está, sin copiarlo dos veces.' : 'También puedes elegirlo.'}
              </p>
            </div>
            <button
              type="button"
              className="btn btn-ghost shrink-0"
              onClick={() => fileInput.current?.click()}
              disabled={busy}
            >
              Elegir
            </button>
            <input
              ref={fileInput}
              type="file"
              accept="video/*,.mkv,.mov,.mp4,.webm"
              className="hidden"
              onChange={(e) => {
                const picked = e.target.files?.[0]
                if (picked) setFile(picked)
              }}
            />
          </div>

          <div className="mt-10 grid gap-8 sm:grid-cols-2">
            <label className="block">
              <span className="eyebrow">Quién puntúa</span>
              <select
                className="field mt-2"
                value={provider}
                onChange={(e) => setProvider(e.target.value)}
                disabled={busy}
              >
                {PROVIDERS.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.label}
                  </option>
                ))}
              </select>
            </label>
            <label className="block">
              <span className="eyebrow">Modelo</span>
              <input
                className="field mt-2"
                placeholder={provider === 'openai' ? 'gpt-4o o el id de tu cuenta' : 'el de Gemma, si no lo cambias'}
                value={model}
                onChange={(e) => setModel(e.target.value)}
                disabled={busy}
                spellCheck={false}
              />
            </label>
            <label className="block">
              <span className="eyebrow">Encuadre</span>
              <select
                className="field mt-2"
                value={layout}
                onChange={(e) => setLayout(e.target.value)}
                disabled={busy}
              >
                {LAYOUTS.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.label}
                  </option>
                ))}
              </select>
            </label>
            <label className="block">
              <span className="eyebrow">Idioma</span>
              <input
                className="field mt-2"
                value={lang}
                onChange={(e) => setLang(e.target.value)}
                disabled={busy}
              />
            </label>
          </div>

          <div className="mt-8 flex flex-wrap items-end gap-6">
            <NumberField label="Clips" value={top} onChange={setTop} disabled={busy} />
            <NumberField label="Mín s" value={minS} onChange={setMinS} disabled={busy} />
            <NumberField label="Máx s" value={maxS} onChange={setMaxS} disabled={busy} />
            <button
              type="button"
              className="btn btn-primary"
              disabled={busy || (!url.trim() && !file)}
              onClick={() => void start()}
            >
              {busy ? 'Arrancando…' : 'Empezar'}
            </button>
          </div>

          {error && (
            <div className="mt-6">
              <ErrorNote error={error} />
            </div>
          )}
        </section>

        <section className="rise" style={{ animationDelay: '160ms' }}>
          <p className="eyebrow">Corridas</p>
          {runs.length === 0 ? (
            <p className="mt-4 text-sm text-ink-500">Todavía no hay ninguna.</p>
          ) : (
            <ul className="mt-4 divide-y divide-ink-800">
              {runs.map((run) => (
                <li key={run.id}>
                  <button
                    type="button"
                    onClick={() => navigate(`/runs/${run.id}`)}
                    className="flex w-full items-baseline justify-between gap-4 py-3 text-left"
                  >
                    <span className="min-w-0">
                      <span className="block truncate text-sm text-ink-100">{run.title}</span>
                      <span className="mt-1 block text-xs text-ink-500">
                        {run.provider} · {statusLabel(run.status)}
                      </span>
                    </span>
                    <span className="numeric text-xs text-ink-400">
                      {run.clips[0] ? formatDuration(run.clips.reduce((s, c) => s + c.duration, 0)) : ''}
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </section>
      </div>
    </div>
  )
}

function NumberField({
  label,
  value,
  onChange,
  disabled,
}: {
  label: string
  value: number
  onChange: (value: number) => void
  disabled: boolean
}) {
  return (
    <label className="block w-20">
      <span className="eyebrow">{label}</span>
      <input
        className="field numeric mt-2"
        type="number"
        value={value}
        min={1}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </label>
  )
}

function statusLabel(status: RawRun['status']): string {
  if (status === 'done') return 'listo'
  if (status === 'failed') return 'falló'
  return 'en curso'
}
