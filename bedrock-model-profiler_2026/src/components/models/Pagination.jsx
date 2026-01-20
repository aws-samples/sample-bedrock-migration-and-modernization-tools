import { ChevronLeft, ChevronRight, ChevronsLeft, ChevronsRight } from 'lucide-react'
import { Button } from '@/components/ui/button'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { useTheme } from '@/components/layout/ThemeProvider'
import { cn } from '@/lib/utils'

export function Pagination({
  currentPage,
  totalPages,
  pageSize,
  totalItems,
  pageSizeOptions = [10, 20, 50],
  onPageChange,
  onPageSizeChange,
}) {
  const { theme } = useTheme()
  const isLight = theme === 'light'

  const startItem = (currentPage - 1) * pageSize + 1
  const endItem = Math.min(currentPage * pageSize, totalItems)

  return (
    <div className="flex flex-col sm:flex-row items-center justify-between gap-3 py-4">
      {/* Items info */}
      <div className={cn('text-sm text-center sm:text-left', isLight ? 'text-slate-500' : 'text-slate-400')}>
        <span className="hidden sm:inline">Showing </span>
        <span className={cn('font-medium', isLight ? 'text-slate-900' : 'text-white')}>{startItem}</span>
        <span className="hidden sm:inline"> to </span>
        <span className="sm:hidden">-</span>
        <span className={cn('font-medium', isLight ? 'text-slate-900' : 'text-white')}>{endItem}</span>
        {' '}of{' '}
        <span className={cn('font-medium', isLight ? 'text-slate-900' : 'text-white')}>{totalItems}</span>
        <span className="hidden sm:inline"> models</span>
      </div>

      {/* Controls */}
      <div className="flex items-center gap-2 sm:gap-4 flex-wrap justify-center">
        {/* Page size selector - hidden on mobile */}
        <div className="hidden sm:flex items-center gap-2">
          <span className={cn('text-sm', isLight ? 'text-slate-500' : 'text-slate-400')}>Per page:</span>
          <Select
            value={pageSize.toString()}
            onValueChange={(value) => onPageSizeChange(parseInt(value, 10))}
          >
            <SelectTrigger className="w-20 h-8 text-sm">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {pageSizeOptions.map(size => (
                <SelectItem key={size} value={size.toString()}>
                  {size}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* Page navigation */}
        <div className="flex items-center gap-1">
          {/* First page - hidden on mobile */}
          <Button
            variant="outline"
            size="icon"
            className="h-8 w-8 hidden sm:flex"
            onClick={() => onPageChange(1)}
            disabled={currentPage === 1}
          >
            <ChevronsLeft className="h-4 w-4" />
          </Button>
          <Button
            variant="outline"
            size="icon"
            className="h-8 w-8"
            onClick={() => onPageChange(currentPage - 1)}
            disabled={currentPage === 1}
          >
            <ChevronLeft className="h-4 w-4" />
          </Button>

          {/* Page indicator */}
          <div className="flex items-center gap-1 px-1 sm:px-2">
            <span className={cn('text-sm hidden sm:inline', isLight ? 'text-slate-500' : 'text-slate-400')}>Page</span>
            <Select
              value={currentPage.toString()}
              onValueChange={(value) => onPageChange(parseInt(value, 10))}
            >
              <SelectTrigger className="w-14 sm:w-16 h-8 text-sm">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {Array.from({ length: totalPages }, (_, i) => i + 1).map(page => (
                  <SelectItem key={page} value={page.toString()}>
                    {page}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <span className={cn('text-sm', isLight ? 'text-slate-500' : 'text-slate-400')}>
              <span className="hidden sm:inline">of </span>
              <span className="sm:hidden">/</span>
              {totalPages}
            </span>
          </div>

          <Button
            variant="outline"
            size="icon"
            className="h-8 w-8"
            onClick={() => onPageChange(currentPage + 1)}
            disabled={currentPage === totalPages}
          >
            <ChevronRight className="h-4 w-4" />
          </Button>
          {/* Last page - hidden on mobile */}
          <Button
            variant="outline"
            size="icon"
            className="h-8 w-8 hidden sm:flex"
            onClick={() => onPageChange(totalPages)}
            disabled={currentPage === totalPages}
          >
            <ChevronsRight className="h-4 w-4" />
          </Button>
        </div>
      </div>
    </div>
  )
}
