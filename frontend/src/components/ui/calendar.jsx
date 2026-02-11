import * as React from "react"
import { DayPicker } from "react-day-picker"
import { ChevronLeft, ChevronRight } from "lucide-react"
import { cn } from "@/lib/utils"
import { useTheme } from "@/components/layout/ThemeProvider"

function Calendar({ className, classNames, ...props }) {
  const { theme } = useTheme()
  const isLight = theme === "light"

  return (
    <DayPicker
      className={cn("p-3", className)}
      classNames={{
        months: "flex flex-col sm:flex-row gap-2",
        month: "flex flex-col gap-4",
        month_caption: "flex justify-center pt-1 relative items-center h-7",
        caption_label: cn("text-sm font-medium", isLight ? "text-stone-800" : "text-white"),
        nav: "flex items-center gap-1",
        button_previous: cn(
          "absolute left-1 top-0 inline-flex items-center justify-center rounded-lg h-7 w-7 bg-transparent p-0 opacity-50 hover:opacity-100 transition-colors",
          isLight ? "hover:bg-stone-100" : "hover:bg-white/[0.06]"
        ),
        button_next: cn(
          "absolute right-1 top-0 inline-flex items-center justify-center rounded-lg h-7 w-7 bg-transparent p-0 opacity-50 hover:opacity-100 transition-colors",
          isLight ? "hover:bg-stone-100" : "hover:bg-white/[0.06]"
        ),
        month_grid: "w-full border-collapse",
        weekdays: "flex",
        weekday: cn(
          "w-8 font-medium text-[0.8rem] text-center",
          isLight ? "text-stone-400" : "text-[#6d6e72]"
        ),
        week: "flex w-full mt-1",
        day: "relative p-0 text-center text-sm focus-within:relative focus-within:z-20",
        day_button: cn(
          "inline-flex items-center justify-center rounded-lg h-8 w-8 p-0 font-normal transition-colors cursor-pointer",
          isLight
            ? "text-stone-700 hover:bg-stone-100 focus-visible:ring-1 focus-visible:ring-amber-500"
            : "text-[#e4e5e7] hover:bg-white/[0.06] focus-visible:ring-1 focus-visible:ring-emerald-500"
        ),
        selected: cn(
          isLight
            ? "[&_button]:bg-amber-600 [&_button]:text-white [&_button]:hover:bg-amber-700 [&_button]:rounded-lg"
            : "[&_button]:bg-emerald-500 [&_button]:text-white [&_button]:hover:bg-emerald-600 [&_button]:rounded-lg"
        ),
        today: cn(
          isLight
            ? "[&_button]:bg-amber-50 [&_button]:text-amber-700 [&_button]:font-semibold"
            : "[&_button]:bg-white/[0.06] [&_button]:text-emerald-400 [&_button]:font-semibold"
        ),
        outside: "[&_button]:text-stone-400/50 [&_button]:opacity-50",
        disabled: "[&_button]:text-stone-400/40 [&_button]:opacity-40 [&_button]:cursor-not-allowed",
        range_middle: cn(
          isLight
            ? "[&_button]:bg-amber-50 [&_button]:text-amber-800 [&_button]:rounded-none [&_button]:hover:bg-amber-100"
            : "[&_button]:bg-emerald-500/15 [&_button]:text-emerald-300 [&_button]:rounded-none [&_button]:hover:bg-emerald-500/25"
        ),
        range_start: "[&_button]:rounded-l-lg [&_button]:rounded-r-none",
        range_end: "[&_button]:rounded-r-lg [&_button]:rounded-l-none",
        hidden: "invisible",
        ...classNames,
      }}
      components={{
        Chevron: ({ orientation }) =>
          orientation === "left"
            ? <ChevronLeft className="h-4 w-4" />
            : <ChevronRight className="h-4 w-4" />,
      }}
      {...props}
    />
  )
}

Calendar.displayName = "Calendar"

export { Calendar }
