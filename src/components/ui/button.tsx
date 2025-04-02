import React, { ButtonHTMLAttributes, ReactNode } from 'react';
import { cn } from '@/lib/utils'; // Assumed utility, adjust if needed

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
    variant?: 'default' | 'ghost';
    className?: string;
    children?: ReactNode;
}

export const Button: React.FC<ButtonProps> = ({
    variant = 'default',
    className,
    children,
    ...props
}) => {
    const baseClasses = cn(
        'inline-flex items-center justify-center rounded-md font-semibold transition-colors duration-200',
        'focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2',
        props.disabled && 'opacity-50 cursor-not-allowed',
        variant === 'default' &&
            'bg-blue-500 text-white hover:bg-blue-600', // Changed from primary
        variant === 'ghost' &&
            'text-gray-300 hover:bg-gray-700 hover:text-white', // Changed from secondary and adjusted
        className
    );

    return (
        <button className={baseClasses} {...props}>
            {children}
        </button>
    );
};

