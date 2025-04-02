import React, { TextareaHTMLAttributes } from 'react';
import { cn } from '@/lib/utils'; // Assumed utility, adjust if needed

interface TextareaProps extends TextareaHTMLAttributes<HTMLTextAreaElement> {
    className?: string;
}

export const Textarea: React.FC<TextareaProps> = ({ className, ...props }) => {
    const baseClasses = cn(
        'flex w-full rounded-md border border-input bg-background px-3 py-2 text-sm',
        'placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2',
        'focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50',
        'resize-none', // Added to prevent the user from resizing
        className
    );
    return <textarea className={baseClasses} {...props} />;
};