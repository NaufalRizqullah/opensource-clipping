import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { Maximize2, Pause, Play, RotateCcw, Volume2, VolumeX, Loader2 } from 'lucide-react'
import { formatTimecode } from '../lib/format'
import { useHotkeys } from '../lib/hooks'

const RATES = [0.75, 1, 1.25, 1.5, 2]

const VideoPlayer = forwardRef(function VideoPlayer({ src, poster, ratio = '9:16' }, ref) {
  const video = useRef(null)
  const wrap = useRef(null)
  const [playing, setPlaying] = useState(false)
  const [time, setTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [buffered, setBuffered] = useState(0)
  const [muted, setMuted] = useState(false)
  const [volume, setVolume] = useState(1)
  const [rate, setRate] = useState(1)
  const [waiting, setWaiting] = useState(false)
  const [error, setError] = useState(false)
  const [flash, setFlash] = useState(null)
  const [scrubbing, setScrubbing] = useState(false)
  const [idle, setIdle] = useState(false)
  const idleTimer = useRef(null)

  useImperativeHandle(ref, () => ({ seek: (t) => { if (video.current) video.current.currentTime = t } }))

  useEffect(() => { setError(false); setPlaying(false); setTime(0) }, [src])

  const toggle = useCallback(() => {
    const v = video.current
    if (!v) return
    if (v.paused) { v.play().catch(() => {}); setFlash('play') } else { v.pause(); setFlash('pause') }
  }, [])
  const seekBy = useCallback((d) => {
    const v = video.current
    if (v) v.currentTime = Math.max(0, Math.min(v.duration || 0, v.currentTime + d))
  }, [])
  const fullscreen = () => {
    const el = wrap.current
    if (!el) return
    if (document.fullscreenElement) document.exitFullscreen?.()
    else (el.requestFullscreen || el.webkitRequestFullscreen)?.call(el)
  }
  const toggleMute = () => { const v = video.current; if (v) { v.muted = !v.muted; setMuted(v.muted) } }

  useHotkeys({
    ' ': toggle, k: toggle,
    j: () => seekBy(-5), l: () => seekBy(5),
    ArrowLeft: () => seekBy(-1), ArrowRight: () => seekBy(1),
    m: toggleMute, f: fullscreen,
  })

  useEffect(() => { if (!flash) return; const t = setTimeout(() => setFlash(null), 450); return () => clearTimeout(t) }, [flash])

  const poke = () => {
    setIdle(false)
    clearTimeout(idleTimer.current)
    idleTimer.current = setTimeout(() => setIdle(true), 2200)
  }
  useEffect(() => () => clearTimeout(idleTimer.current), [])

  const scrubTo = (clientX, el) => {
    const rect = el.getBoundingClientRect()
    const pct = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width))
    if (video.current && duration) { video.current.currentTime = pct * duration; setTime(pct * duration) }
  }

  const [rw, rh] = ratio.split(':').map(Number)
  const pct = duration ? (time / duration) * 100 : 0

  return (
    <div
      ref={wrap}
      className={`player ${playing && idle && !scrubbing ? 'is-idle' : ''}`}
      style={{ '--ar': `${rw || 9} / ${rh || 16}` }}
      onMouseMove={poke}
      onMouseLeave={() => setIdle(true)}
    >
      <video
        ref={video}
        src={src}
        poster={poster}
        playsInline
        preload="metadata"
        onClick={toggle}
        onPlay={() => { setPlaying(true); poke() }}
        onPause={() => setPlaying(false)}
        onTimeUpdate={(e) => !scrubbing && setTime(e.currentTarget.currentTime)}
        onLoadedMetadata={(e) => setDuration(e.currentTarget.duration)}
        onDurationChange={(e) => setDuration(e.currentTarget.duration)}
        onProgress={(e) => {
          const b = e.currentTarget.buffered
          if (b.length && e.currentTarget.duration) setBuffered((b.end(b.length - 1) / e.currentTarget.duration) * 100)
        }}
        onWaiting={() => setWaiting(true)}
        onPlaying={() => setWaiting(false)}
        onCanPlay={() => setWaiting(false)}
        onError={() => setError(true)}
        onVolumeChange={(e) => { setMuted(e.currentTarget.muted); setVolume(e.currentTarget.volume) }}
      />

      <AnimatePresence>
        {flash && (
          <motion.div key={flash} className="player-flash" initial={{ opacity: 0.9, scale: 0.8 }} animate={{ opacity: 0, scale: 1.3 }} exit={{ opacity: 0 }} transition={{ duration: 0.45 }}>
            {flash === 'play' ? <Play size={28} fill="currentColor" /> : <Pause size={28} fill="currentColor" />}
          </motion.div>
        )}
      </AnimatePresence>
      {waiting && !error && <div className="player-center"><Loader2 className="spin" size={28} /></div>}
      {error && <div className="player-center player-error">Video unavailable. The file may have been removed or the link expired — reload the page.</div>}
      {!playing && !error && !waiting && (
        <button type="button" className="player-big-play" onClick={toggle} aria-label="Play"><Play size={26} fill="currentColor" /></button>
      )}

      <div className="player-controls" onMouseDown={(e) => e.stopPropagation()}>
        <div
          className="player-scrub"
          role="slider"
          tabIndex={0}
          aria-label="Seek"
          aria-valuemin={0}
          aria-valuemax={Math.round(duration)}
          aria-valuenow={Math.round(time)}
          aria-valuetext={`${formatTimecode(time)} of ${formatTimecode(duration)}`}
          onPointerDown={(e) => { e.currentTarget.setPointerCapture(e.pointerId); setScrubbing(true); scrubTo(e.clientX, e.currentTarget) }}
          onPointerMove={(e) => scrubbing && scrubTo(e.clientX, e.currentTarget)}
          onPointerUp={() => setScrubbing(false)}
          onKeyDown={(e) => { if (e.key === 'ArrowLeft') { e.stopPropagation(); seekBy(-1) } if (e.key === 'ArrowRight') { e.stopPropagation(); seekBy(1) } }}
        >
          <div className="scrub-track">
            <div className="scrub-buffer" style={{ width: `${buffered}%` }} />
            <div className="scrub-fill" style={{ width: `${pct}%` }} />
            <div className="scrub-thumb" style={{ left: `${pct}%` }} />
          </div>
        </div>
        <div className="player-row">
          <button type="button" className="player-btn" onClick={toggle} aria-label={playing ? 'Pause' : 'Play'}>{playing ? <Pause size={18} fill="currentColor" /> : <Play size={18} fill="currentColor" />}</button>
          <button type="button" className="player-btn" onClick={() => { if (video.current) video.current.currentTime = 0 }} aria-label="Restart"><RotateCcw size={16} /></button>
          <div className="player-volume">
            <button type="button" className="player-btn" onClick={toggleMute} aria-label={muted ? 'Unmute' : 'Mute'}>{muted || volume === 0 ? <VolumeX size={17} /> : <Volume2 size={17} />}</button>
            <input type="range" min="0" max="1" step="0.05" value={muted ? 0 : volume} aria-label="Volume" onChange={(e) => { const v = video.current; if (v) { v.volume = Number(e.target.value); v.muted = Number(e.target.value) === 0 } }} style={{ '--pct': `${(muted ? 0 : volume) * 100}%` }} />
          </div>
          <span className="player-time mono">{formatTimecode(time)} / {formatTimecode(duration)}</span>
          <span className="grow" />
          <button type="button" className="player-btn player-rate" onClick={() => { const next = RATES[(RATES.indexOf(rate) + 1) % RATES.length]; setRate(next); if (video.current) video.current.playbackRate = next }} aria-label={`Playback speed ${rate}x`}>{rate}×</button>
          <button type="button" className="player-btn" onClick={fullscreen} aria-label="Fullscreen"><Maximize2 size={16} /></button>
        </div>
      </div>
    </div>
  )
})

export default VideoPlayer
