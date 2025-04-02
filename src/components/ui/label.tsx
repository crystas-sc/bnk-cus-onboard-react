import React, { LabelHTMLAttributes, ReactNode } from 'react';
import { cn } from '@/lib/utils';  // Assumed utility, adjust if needed

interface LabelProps extends LabelHTMLAttributes<HTMLLabelElement> {
    className?: string;
    children?: ReactNode;
}

export const Label: React.FC<LabelProps> = ({ className, children, ...props }) => {
    const baseClasses = cn(
        'text-sm font-medium leading-none peer-disabled:cursor-not-allowed peer-disabled:opacity-70',
        className
    );
    return <label className={baseClasses} {...props}>{children}</label>;
};