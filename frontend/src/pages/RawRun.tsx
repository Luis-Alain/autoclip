import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'

import { api, formatDuration, type RawRun } from '../api'
import { ErrorNote } from '../components/ErrorNote'

const LAYOUT_LABEL: Record<string, string> = {
  face: 'Persona',
  split: 'Dividida',
  blur: 'Blur',
}

export function RawRunPage() {
  const { runId } = useParams()
  const [run, setRun] = useState<RawRun | null>(null)
  const [error, setError] = useState<Error | null>(null)

  useEffect(() => {
    if (!runId) return
    let stop = false

    const pull = () => {
      api
        .getRawRun(runId)
        .then((next) => {
          if (!stop) setRun(next)
        })
        .catch((err: Error) => {
          if (!stop) setError(err)
        })
    }

    pull()
    const timer = window.setInterval(pull, 2000)
    return () => {
      stop = true
      window.clearInterval(timer)
    }
  }, [runId])

  if (error && !run) {
    return (
      <div className="pt-14">
        <ErrorNote error={error} />
      </div>
    )
  }
  if (!run) {
    return <p className="pt-24 text-sm text-ink-500">Cargando…</p>
  }

  const percent = Math.round(run.progress * 100)

  return (
    <div className="pt-14">
      <div className="rise flex flex-wrap items-baseline justify-between gap-6 border-b border-ink-800 pb-6">
        <div className="min-w-0">
          <p className="eyebrow">{statusLabel(run.status)}</p>
          <h1 className="mt-2 max-w-3xl truncate font-display text-[clamp(1.75rem,4vw,3rem)] leading-tight text-ink-100">
            {run.title}
          </h1>
          <p className="mt-2 text-xs text-ink-500">
            {run.provider}
            {run.layout ? ` · ${run.layout}` : ''}
            {run.message ? ` · ${run.message}` : ''}
          </p>
        </div>
        <span className="numeric font-display text-[clamp(2.5rem,6vw,4rem)] leading-none text-sodium-500">
          {percent}
          <span className="text-2xl text-ink-600">%</span>
        </span>
      </div>

      <div className="mt-4 h-px w-full bg-ink-800">
        <div className="h-px bg-sodium-500" style={{ width: `${percent}%` }} />
      </div>

      {run.error && (
        <div className="mt-8">
          <ErrorNote error={new Error(run.error)} />
          <Link to="/" className="btn btn-ghost mt-4">
            Volver
          </Link>
        </div>
      )}

      {run.status === 'done' && (
        <section className="mt-12">
          <div className="flex flex-wrap items-baseline justify-between gap-4">
            <p className="eyebrow">{run.clips.length} clips en 9:16</p>
            {run.timeline_url && (
              <a className="btn btn-ghost" href={run.timeline_url}>
                timeline.xml
              </a>
            )}
          </div>
          {run.clips.length === 0 ? (
            <p className="mt-6 text-sm text-ink-400">
              El modelo no se quedó con ningún momento. Prueba otro modelo o un rango de duración más amplio.
            </p>
          ) : (
            <ul className="mt-8 grid gap-10 sm:grid-cols-2 xl:grid-cols-3">
              {run.clips.map((clip) => (
                <li key={clip.file}>
                  <video
                    className="aspect-[9/16] w-full bg-ink-850"
                    src={clip.url}
                    controls
                    preload="metadata"
                  />
                  <div className="mt-3 flex items-baseline justify-between gap-3">
                    <h2 className="font-display text-xl leading-tight text-ink-100">
                      {clip.hook_title}
                    </h2>
                    <span className="numeric text-sm text-sodium-500">{clip.score}</span>
                  </div>
                  <p className="mt-1 text-xs text-ink-500">
                    {formatDuration(clip.duration)} · {LAYOUT_LABEL[clip.layout] ?? clip.layout}
                  </p>
                  {clip.reason && <p className="mt-2 text-sm leading-relaxed text-ink-300">{clip.reason}</p>}
                </li>
              ))}
            </ul>
          )}
        </section>
      )}
    </div>
  )
}

function statusLabel(status: RawRun['status']): string {
  if (status === 'done') return 'listo'
  if (status === 'failed') return 'falló'
  return 'en curso'
}
