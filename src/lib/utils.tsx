import { twMerge } from 'tailwind-merge';
import { clsx, ClassValue } from 'clsx';

/**
 * Merges and deduplicates CSS class names.
 *
 * It uses clsx to handle conditional classes and tailwind-merge to
 * ensure that Tailwind CSS classes are correctly combined.
 *
 * @param inputs - Any number of class name arguments.
 * Can be strings, objects (for conditional classes),
 * or arrays of class names.
 * @returns A string with combined and deduplicated class names.
 */
export function cn(...inputs: ClassValue[]): string {
    return twMerge(clsx(inputs));
}