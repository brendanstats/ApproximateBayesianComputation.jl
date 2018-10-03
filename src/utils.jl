"""
Converts a Base.Dates.XX format into a string in HH:MM:SS format.
"""
function duration_to_string(duration::Base.Dates.Hour)
    duration = Dates.value(duration)
    return string(duration)
end

function duration_to_string(duration::Base.Dates.Minute)
    duration = Dates.value(duration)
    minutes = duration % 60
    hours = (duration - minutes) / 60
    minutes = Dates.value(minutes)
    hours = Dates.value(hours)
    return string(hours, ":", minutes)
end

function duration_to_string(duration::Base.Dates.Second)
    duration = Dates.value(duration)
    seconds = duration % 60
    duration = (duration - seconds) / 60
    minutes = duration % 60
    hours = (duration - minutes) / 60
    seconds = convert(Int64, seconds)
    minutes = convert(Int64, minutes)
    hours = convert(Int64, hours)
    return string(hours, ":", minutes, ":", seconds)
end

function duration_to_string(duration::Base.Dates.Millisecond)
    duration = Dates.value(duration)
    duration = floor(duration / 1000)
    seconds = duration % 60
    duration = (duration - seconds) / 60
    minutes = duration % 60
    hours = (duration - minutes) / 60
    seconds = convert(Int64, seconds)
    minutes = convert(Int64, minutes)
    hours = convert(Int64, hours)
    return string(hours, ":", minutes, ":", seconds)
end

"""
Returns only the rows (2D) or entries corresponding to the supplied indicies
"""
function subset(A::Array{T, 1}, idxs::Array{G}) where {T <: Any, G <: Integer}
    return A[idxs]
end

function subset!(A::Array{T, 1}, idxs::Array{G}) where {T <: Any, G <: Integer}
    A = A[idxs]
    nothing
end

function subset(A::Array{T, 2}, idxs::Array{G}) where {T <: Any, G <: Integer}
    return A[idxs, :]
end

function subset!(A::Array{T, 2}, idxs::Array{G}) where {T <: Any, G <: Integer}
    A = A[idxs, :]
    nothing
end
