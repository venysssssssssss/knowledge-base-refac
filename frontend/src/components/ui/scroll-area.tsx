import React, { ReactNode, CSSProperties } from 'react'

interface ScrollAreaProps {
  children: ReactNode
  className?: string
  style?: CSSProperties
}

export function ScrollArea({ children, className = '', style }: ScrollAreaProps) {
  return (
    <div
      className={`overflow-y-auto scrollbar-thin scrollbar-thumb-gray-400 scrollbar-track-gray-200 dark:scrollbar-thumb-gray-600 dark:scrollbar-track-gray-700 ${className}`}
      style={style}
    >
      {children}
    </div>
  )
}
