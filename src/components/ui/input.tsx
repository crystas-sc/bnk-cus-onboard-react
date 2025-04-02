import React, { InputHTMLAttributes } from 'react';
import { cn } from '@/lib/utils'; // Assumed utility, adjust if needed

interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
    className?: string;
}

export const Input: React.FC<InputProps> = ({ className, ...props }) => {
    const baseClasses = cn(
        'flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm',
        'file:border-0 file:bg-transparent file:text-sm file:font-medium',
        'placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2',
        'focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50',
        className
    );

    return <input className={baseClasses} {...props} />;
};