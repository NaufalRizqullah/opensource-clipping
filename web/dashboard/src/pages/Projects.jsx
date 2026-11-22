import { useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { AnimatePresence } from 'motion/react'
import { FolderKanban, Plus, Search, RefreshCw } from 'lucide-react'
import { api } from '../lib/api'
import { useDocumentTitle, useLocalState, useResource } from '../lib/hooks'
import { isActive, jobTitle } from '../lib/format'
import { PageHeader } from '../components/Layout'
import { Button, EmptyState, IconButton, Input, Segmented, Select } from '../components/ui'
import { ProjectCard, ProjectCardSkeleton } from '../components/project'

const FILTERS = [
  { value: 'all', label: 'All' },
  { value: 'active', label: 'Active' },
  { value: 'completed', label: 'Completed' },
  { value: 'failed', label: 'Failed' },
]

export default function Projects() {
  useDocumentTitle('Projects')
  const navigate = useNavigate()
  const [filter, setFilter] = useLocalState('osc.projects.filter', 'all')
  const [sort, setSort] = useLocalState('osc.projects.sort', 'newest')
  const [query, setQuery] = useState('')
  const { data, error, loading, reload } = useResource((signal) => api.jobs({ signal }), [], {
    interval: (d) => (d?.jobs?.some(isActive) ? 4000 : 15000),
  })

  const jobs = useMemo(() => {
    let list = data?.jobs || []
    if (filter === 'active') list = list.filter(isActive)
    else if (filter === 'completed') list = list.filter((j) => j.status === 'completed')
    else if (filter === 'failed') list = list.filter((j) => j.status === 'failed' || j.status === 'cancelled')
    const q = query.trim().toLowerCase()
    if (q) {
      list = list.filter((j) => [jobTitle(j), j.id, j.url, j.upload_filename, ...(j.clips || []).map((c) => c.title_en || c.title)]
        .filter(Boolean).some((s) => String(s).toLowerCase().includes(q)))
    }
    const sorted = [...list]
    if (sort === 'oldest') sorted.sort((a, b) => new Date(a.created_at) - new Date(b.created_at))
    else if (sort === 'clips') sorted.sort((a, b) => (b.clips?.length || 0) - (a.clips?.length || 0))
    else sorted.sort((a, b) => new Date(b.created_at) - new Date(a.created_at))
    return sorted
  }, [data, filter, query, sort])

  const total = data?.jobs?.length || 0

  return (
    <>
      <PageHeader
        title="Projects"
        description={total ? `${total} project${total === 1 ? '' : 's'} on this server` : 'Every clipping job lives here.'}
        actions={
          <>
            <IconButton icon={RefreshCw} label="Refresh" onClick={reload} />
            <Button variant="primary" icon={Plus} onClick={() => navigate('/new')}>New project</Button>
          </>
        }
      />

      <div className="toolbar">
        <Input icon={Search} type="search" placeholder="Search titles, links, IDs…" value={query} onChange={(e) => setQuery(e.target.value)} aria-label="Search projects" className="toolbar-search" />
        <Segmented value={filter} onChange={setFilter} options={FILTERS} label="Filter by status" layoutId="projects-filter" />
        <Select value={sort} onChange={(e) => setSort(e.target.value)} aria-label="Sort projects" className="toolbar-sort">
          <option value="newest">Newest first</option>
          <option value="oldest">Oldest first</option>
          <option value="clips">Most clips</option>
        </Select>
      </div>

      {error && <p className="alert alert-danger">{error.message}</p>}

      {loading && !data ? (
        <div className="project-grid">{Array.from({ length: 6 }, (_, i) => <ProjectCardSkeleton key={i} />)}</div>
      ) : jobs.length ? (
        <div className="project-grid">
          <AnimatePresence mode="popLayout">
            {jobs.map((job, i) => <ProjectCard key={job.id} job={job} index={i} />)}
          </AnimatePresence>
        </div>
      ) : (
        <EmptyState
          icon={FolderKanban}
          title={total ? 'No matching projects' : 'No projects yet'}
          description={total ? 'Try a different search or filter.' : 'Start a project from a video link or an uploaded file.'}
          action={!total && <Button variant="primary" icon={Plus} onClick={() => navigate('/new')}>New project</Button>}
        />
      )}
    </>
  )
}
