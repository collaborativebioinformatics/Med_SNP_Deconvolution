#!/usr/bin/env python3
"""
Demonstration of the label skew algorithm logic without dependencies
This shows how classes are distributed to sites
"""
import numpy as np


def demonstrate_label_skew_distribution(num_classes, num_sites, labels_per_site):
    """
    Demonstrates how classes are distributed to sites using the label skew algorithm

    Args:
        num_classes: Number of classes in the dataset
        num_sites: Number of federated sites
        labels_per_site: Number of labels each site should get
    """
    print("\n" + "="*80)
    print(f"Label Skew Distribution Demo")
    print(f"  Classes: {num_classes}, Sites: {num_sites}, Labels per site: {labels_per_site}")
    print("="*80)

    unique_labels = np.arange(num_classes)

    # Adjust labels_per_site if needed
    if labels_per_site > num_classes:
        print(f"WARNING: labels_per_site ({labels_per_site}) > num_classes ({num_classes})")
        print(f"         Adjusting to {num_classes}")
        labels_per_site = num_classes

    # Assign classes to sites using the sliding window approach
    site_labels = {}

    if labels_per_site >= num_classes:
        # All sites get all classes
        for site_id in range(num_sites):
            site_labels[site_id] = list(unique_labels)
    else:
        # Use sliding window with overlap
        # Calculate optimal step size to ensure coverage and moderate overlap
        total_label_slots = num_sites * labels_per_site

        # Calculate maximum classes that can be covered with current settings
        # With step size s, we cover: labels_per_site + (num_sites - 1) * s
        # We want this to be >= num_classes
        #
        # Simple heuristic: if slots allow coverage with overlap, use smaller step
        # Otherwise use larger step to maximize coverage

        # Option 1: step that allows overlap (step = labels_per_site // 2)
        step_with_overlap = max(1, labels_per_site // 2) if labels_per_site > 1 else 1
        max_coverage_with_overlap = labels_per_site + (num_sites - 1) * step_with_overlap

        if max_coverage_with_overlap >= num_classes:
            # Can cover all classes with overlap
            step = step_with_overlap
        else:
            # Need larger step to cover more classes
            # Calculate minimum step needed
            if num_sites == 1:
                step = 1
            else:
                min_step = max(1, (num_classes - labels_per_site + num_sites - 2) // (num_sites - 1))
                step = min_step
            print(f"  Note: Using step size {step} to maximize class coverage")

        for site_id in range(num_sites):
            start_idx = (site_id * step) % num_classes
            selected_labels = []
            for i in range(labels_per_site):
                label_idx = (start_idx + i) % num_classes
                selected_labels.append(unique_labels[label_idx])
            site_labels[site_id] = selected_labels

    # Print the distribution
    print("\nClass Distribution to Sites:")
    print("-" * 80)
    for site_id in range(num_sites):
        labels = site_labels[site_id]
        print(f"  Site {site_id + 1}: Classes {labels}")

    # Verify all classes are covered
    all_assigned = set()
    for labels in site_labels.values():
        all_assigned.update(labels)

    print("\nCoverage Analysis:")
    print("-" * 80)
    print(f"  Total unique classes assigned: {len(all_assigned)}")
    print(f"  All classes covered: {len(all_assigned) == num_classes}")

    if len(all_assigned) < num_classes:
        missing = set(unique_labels) - all_assigned
        print(f"  WARNING: Missing classes: {missing}")

    # Count how many sites share each class
    class_site_count = {label: 0 for label in unique_labels}
    for labels in site_labels.values():
        for label in labels:
            class_site_count[label] += 1

    print("\nClass Sharing Statistics:")
    print("-" * 80)
    for label, count in sorted(class_site_count.items()):
        percentage = (count / num_sites) * 100
        print(f"  Class {label}: shared by {count}/{num_sites} sites ({percentage:.1f}%)")

    avg_sharing = np.mean(list(class_site_count.values()))
    print(f"\n  Average sites per class: {avg_sharing:.2f}")
    print(f"  Label skew factor: {1 - (avg_sharing / num_sites):.2%}")
    print("    (0% = no skew/IID, 100% = maximum skew)")


if __name__ == "__main__":
    print("\n" + "#"*80)
    print("# Label Skew Algorithm Demonstration")
    print("#"*80)

    # Scenario 1: Standard case from requirements
    demonstrate_label_skew_distribution(
        num_classes=3,
        num_sites=3,
        labels_per_site=2
    )

    # Scenario 2: More sites than classes
    demonstrate_label_skew_distribution(
        num_classes=3,
        num_sites=5,
        labels_per_site=2
    )

    # Scenario 3: Extreme skew (1 label per site)
    demonstrate_label_skew_distribution(
        num_classes=4,
        num_sites=4,
        labels_per_site=1
    )

    # Scenario 4: Many classes, moderate skew
    demonstrate_label_skew_distribution(
        num_classes=10,
        num_sites=5,
        labels_per_site=3
    )

    # Scenario 5: Edge case - labels_per_site >= num_classes
    demonstrate_label_skew_distribution(
        num_classes=3,
        num_sites=3,
        labels_per_site=5
    )

    print("\n" + "#"*80)
    print("# Demonstration Complete!")
    print("#"*80 + "\n")
