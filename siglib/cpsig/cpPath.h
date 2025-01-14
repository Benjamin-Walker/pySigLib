#pragma once
#include "cppch.h"

//Path has a single iterator, which returns a pointer to an Element class. Element then has its own [], ++, etc operators
//which deal with the given setup.

//have three options for on-the-fly computations: timeaug, leadlag, timeaugleadlag. Every other combination can be done in
//pre-processing for now.

#pragma pack(push)
//#pragma pack(1)

template<typename T>
class PointImpl;

template<typename T>
class PointImplTimeAug;

template<typename T>
class PointImplLeadLag;

template<typename T>
class PointImplTimeAugLeadLag;

template<typename T>
class Point;

template<typename T>
class Path {
public:
	//Add concat, push_back functions
	//Add time reversal function
	//Add time augmentation
	//Add lead lag
	static_assert(std::is_arithmetic<T>::value);

	Path(T* data_, uint64_t dimension_, uint64_t length_, bool timeAug_ = false, bool leadLag_ = false) :
		_dimension{ (leadLag_ ? 2 * dimension_ : dimension_) + (timeAug_ ? 1 : 0) },
		_length{ leadLag_ ? length_ * 2 - 3 : length_ },
		_data{ std::span<T>(data_, dimension_ * length_) },
		_dataDimension{ dimension_ },
		_dataLength{ length_ },
		_dataSize{ dimension_ * length_ },
		_timeAug{ timeAug_ },
		_leadLag{ leadLag_ } {
	}

	Path(const std::span<T> data_, uint64_t dimension_, uint64_t length_, bool timeAug_ = false, bool leadLag_ = false) :
		_dimension{ (leadLag_ ? 2 * dimension_ : dimension_) + (timeAug_ ? 1 : 0) },
		_length{ leadLag_ ? length_ * 2 - 3 : length_ },
		_data{ data_ },
		_dataDimension{ dimension_ },
		_dataLength{ length_ },
		_dataSize{ dimension_ * length_ },
		_timeAug{ timeAug_ },
		_leadLag{ leadLag_ } {
		if (data_.size() != dimension_ * length_)
			throw std::invalid_argument("1D vector is not the correct shape for a path of dimension " + std::to_string(dimension_) + " and length " + std::to_string(length_));
	}

	Path(const Path& other) :
		_dimension{ other._dimension },
		_length{ other._length },
		_data{ other._data },
		_dataDimension{ other._dataDimension },
		_dataLength{ other._dataLength },
		_dataSize{ other._dataSize },
		_timeAug{ other._timeAug },
		_leadLag{ other._leadLag } {
	}

	Path(const Path& other, bool timeAug_, bool leadLag_) :
		_dimension{ (leadLag_ ? 2 * other._dataDimension : other._dataDimension) + (timeAug_ ? 1 : 0) },
		_length{ leadLag_ ? other._dataLength * 2 - 3 : other._dataLength },
		_data{ other._data },
		_dataDimension{ other._dataDimension },
		_dataLength{ other._dataLength },
		_dataSize{ other._dataSize },
		_timeAug{ timeAug_ },
		_leadLag{ leadLag_ } {
	}

	virtual ~Path() {}

	Path<T>& operator=(const Path&) = delete;

	inline uint64_t dimension() const { return _dimension; }
	inline uint64_t length() const { return _length; }
	inline T* data() const { return _data.data(); }

	inline bool timeAug() const { return _timeAug; }
	inline bool leadLag() const { return _leadLag; }

	friend class Point<T>;
	friend class PointImpl<T>;
	friend class PointImplTimeAug<T>;
	friend class PointImplLeadLag<T>;
	friend class PointImplTimeAugLeadLag<T>;

	Point<T> operator[](uint64_t i) const { 
#ifdef _DEBUG
		if (i < 0 || i >= _length)
			throw std::out_of_range("Argument out of bounds in Path::operator[]");
#endif
		return Point<T>(this, i);
	}

	inline Point<T> begin() const
	{
		return Point<T>(this, 0);
	}
	inline Point<T> end() const
	{
		return Point<T>(this, _length); 
	}

	bool operator==(const Path& other) const {
		return _data.data() == other._data.data()
			&& _timeAug == other._timeAug
			&& _leadLag == other._leadLag;
	}
	bool operator!=(const Path& other) const {
		return !this->operator==(other);
	}

	PointImpl<T>* pointImplFactory(uint64_t index) const;

private:
	const uint64_t _dimension;
	const uint64_t _length;

	const std::span<T> _data;
	const uint64_t _dataDimension;
	const uint64_t _dataLength;
	const uint64_t _dataSize;

	const bool _timeAug;
	const bool _leadLag;
};

template<typename T>
class PointImpl {
public:
	PointImpl() : ptr{ nullptr }, path{ nullptr } {}
	PointImpl(const Path<T>* path_, uint64_t index) :
		ptr{ path_->_data.data() + index * path_->_dataDimension },
		path{ path_ }
	{}
	PointImpl(const PointImpl& other) : 
		ptr{ other.ptr },
		path{ other.path }
	{}
	virtual ~PointImpl() {}

	virtual PointImpl<T>* duplicate() const {
		auto p = new PointImpl();
		p->ptr = ptr;
		p->path = path;
		return p;
	}

	virtual inline T operator[](uint64_t i) const { return ptr[i]; } //Change to double
	virtual inline void operator++() { ptr += path->_dataDimension; }
	virtual inline void operator--() { ptr -= path->_dataDimension; }

	inline uint64_t dimension() { return path->_dimension; }
	virtual inline void advance(int64_t n) { ptr += n * path->_dataDimension; }
	virtual inline void setToStart() { ptr = path->_data.data(); }
	virtual inline void setToEnd() { ptr = path->_data.data() + path->_dataSize; }
	virtual inline void setToIndex(int64_t n) { ptr = path->_data.data() + n * path->_dataDimension; }

	inline T* data() const { return ptr; }
	virtual inline uint64_t index() const { return static_cast<uint64_t>((ptr - path->_data.data()) / path->_dataDimension); }

	bool operator==(const PointImpl& other) const { return path == other.path && index() == other.index(); }
	bool operator!=(const PointImpl& other) const { return path != other.path || index() != other.index(); }
	bool operator<(const PointImpl& other) const { return path == other.path && index() < other.index(); }
	bool operator<=(const PointImpl& other) const { return path == other.path && index() <= other.index(); }
	bool operator>(const PointImpl& other) const { return path == other.path && index() > other.index(); }
	bool operator>=(const PointImpl& other) const { return path == other.path && index() >= other.index(); }

	T* ptr;
	const Path<T>* path;
};

template<typename T>
class PointImplTimeAug : public PointImpl<T> {
public:
	PointImplTimeAug() : PointImpl<T>(), time{ 0 } {}
	PointImplTimeAug(const Path<T>* path_, uint64_t index) : PointImpl<T>(path_, index), time{ static_cast<T>(index) } {}
	PointImplTimeAug(const PointImplTimeAug& other) : PointImpl<T>(other), time{ other.time } {}
	virtual ~PointImplTimeAug() {}

	PointImpl<T>* duplicate() const override {
		auto p = new PointImplTimeAug();
		p->ptr = this->ptr;
		p->path = this->path;
		p->time = this->time;
		return p;
	}

	inline T operator[](uint64_t i) const override { return (i < this->path->_dataDimension) ? this->ptr[i] : time;	}
	inline void operator++() override { this->ptr += this->path->_dataDimension; time += 1;	}
	inline void operator--() override { this->ptr -= this->path->_dataDimension; time -= 1; }
	inline void advance(int64_t n) override { this->ptr += n * this->path->_dataDimension; time += static_cast<T>(n); }
	inline void setToStart() override { this->ptr = this->path->_data.data(); time = 0; }
	inline void setToEnd() override { this->ptr = this->path->_data.data() + this->path->_dataSize; time = static_cast<T>(this->path->_length); }
	inline void setToIndex(int64_t n) override { this->ptr = this->path->_data.data() + n * this->path->_dataDimension; time = static_cast<T>(n); }

private:
	T time; //Change time to [0,1]
};

template<typename T>
class PointImplLeadLag : public PointImpl<T> {
public:
	PointImplLeadLag() : PointImpl<T>(), parity{ false } {}
	PointImplLeadLag(const Path<T>* path_, uint64_t index) : PointImpl<T>(path_, index / 2), parity{ static_cast<bool>(index % 2) } {}
	PointImplLeadLag(const PointImplLeadLag& other) : PointImpl<T>(other), parity{ other.parity } {}
	virtual ~PointImplLeadLag() {}

	PointImpl<T>* duplicate() const override {
		auto p = new PointImplLeadLag();
		p->ptr = this->ptr;
		p->path = this->path;
		p->parity = this->parity;
		return p;
	}

	inline T operator[](uint64_t i) const override { 
		if (i < this->path->_dataDimension)
			return this->ptr[i];
		else {
			uint64_t leadIdx = parity ? this->path->_dataDimension + i : i;
			return this->ptr[leadIdx];
		}
	}
	inline void operator++() override { if (parity) this->ptr += this->path->_dataDimension; parity = !parity; }
	inline void operator--() override { if (!parity) this->ptr -= this->path->_dataDimension; parity = !parity; }
	inline void advance(int64_t n) override { this->ptr += (n / 2) * this->path->_dataDimension; parity = (parity != static_cast<bool>(n % 2)); }
	inline void setToStart() override { this->ptr = this->path->_data.data(); parity = false; }
	inline void setToEnd() override { this->ptr = this->path->_data.data() + this->path->_dataSize; parity = true; }
	inline void setToIndex(int64_t n) override { this->ptr = this->path->_data.data() + (n / 2) * this->path->_dataDimension; parity = static_cast<bool>(n % 2); }

	inline uint64_t index() const override { return 2 * static_cast<uint64_t>(this->ptr - this->path->_data.data()) + static_cast<uint64_t>(parity); }

private:
	bool parity;
};

template<typename T>
class PointImplTimeAugLeadLag : public PointImpl<T> {
public:
	PointImplTimeAugLeadLag() : PointImpl<T>(), parity{ false }, time{ 0 }, _dataDimensionTimes2{ 0 } {}
	PointImplTimeAugLeadLag(const Path<T>* path_, uint64_t index) : 
		PointImpl<T>(path_, index / 2), 
		parity{ static_cast<bool>(index % 2) },
		time{ static_cast<T>((index / 2) * 3 + (index % 2) * 2)},
		_dataDimensionTimes2{ path_->_dataDimension * 2 }
	{}
	PointImplTimeAugLeadLag(const PointImplTimeAugLeadLag& other) :
		PointImpl<T>(other), 
		parity{ other.parity },
		time{ other.time },
		_dataDimensionTimes2{ other._dataDimensionTimes2 }{}
	virtual ~PointImplTimeAugLeadLag() {}

	PointImpl<T>* duplicate() const override {
		auto p = new PointImplTimeAugLeadLag();
		p->ptr = this->ptr;
		p->path = this->path;
		p->parity = this->parity;
		p->time = this->time;
		p->_dataDimensionTimes2 = this->_dataDimensionTimes2;
		return p;
	}

	inline T operator[](uint64_t i) const override {
		if (i < this->path->_dataDimension)
			return this->ptr[i];
		else if (i < this->_dataDimensionTimes2) {
			uint64_t lead_idx = parity ? this->path->_dataDimension + i : i;
			return this->ptr[lead_idx];
		}
		else
			return time;
	}
	inline void operator++() override {
		if (parity) { this->ptr += this->path->_dataDimension; time += 1; }
		else time += 2;
		parity = !parity;
	}
	inline void operator--() override {
		if (!parity) { this->ptr -= this->path->_dataDimension; time -= 1; }
		else time -= 2;
		parity = !parity;
	}
	inline void advance(int64_t n) override { 
		this->ptr += (n / 2) * this->path->_dataDimension; 
		parity = (parity != static_cast<bool>(n % 2)); 
		time += (parity) ? static_cast<T>((n / 2) * 3 + (n % 2)) : static_cast<T>((n / 2) * 3 + (n % 2) * 2);
	}
	inline void setToStart() override { this->ptr = this->path->_data.data(); parity = false; time = 0; }
	inline void setToEnd() override { this->ptr = this->path->_data.data() + this->path->_dataSize; parity = true; time = static_cast<T>(this->path->_length); }
	inline void setToIndex(int64_t n) override { 
		this->ptr = this->path->_data.data() + (n / 2) * this->path->_dataDimension;
		parity = static_cast<bool>(n % 2);
		time = static_cast<T>( (n / 2) * 3 + (n % 2) * 2 );
	}

	inline uint64_t index() const override { return 2UL * static_cast<uint64_t>(this->ptr - this->path->_data.data()) + static_cast<uint64_t>(parity); }

private:
	bool parity;
	T time;
	uint64_t _dataDimensionTimes2;
};

template<typename T>
class Point {
public:
	static_assert(std::is_arithmetic<T>::value);

	Point() {
		_impl.reset(nullptr);
	}

	Point(const Path<T>* path, uint64_t index) {
		// Create new impl from path and index - path knows how
		_impl.reset(path->pointImplFactory(index));
	}

	Point(const Point& other) {
		_impl.reset(other._impl->duplicate());
	}

	Point(Point&& other)
	{
		_impl.swap(other._impl);
	}

	Point& operator=(const Point& other) {
		if (this != &other) {
			_impl.reset(other._impl->duplicate());
		}
		return *this;
	}

	Point& operator=(Point&& other) {
		_impl.swap(other._impl);
		return *this;
	}


	inline T operator[](uint64_t i) const { 
#ifdef _DEBUG
		sqBracketBoundsCheck(i);
#endif
		return _impl->operator[](i);
	}
	inline Point& operator++() {
		_impl->operator++();
		return *this;
	}
	inline Point operator++(int) {
		Point tmp{ *this };
		++(*this);
		return tmp;
	}
	inline Point& operator--() {
		_impl->operator--();
		return *this;
	}
	inline Point operator--(int) {
		Point tmp{ *this };
		--(*this);
		return tmp;
	}

	inline uint64_t dimension() { return _impl->dimension(); }
	inline void advance(int64_t n) { 
		_impl->advance(n);
	}
	inline void setToStart() { _impl->setToStart(); }
	inline void setToEnd() { _impl->setToEnd(); }
	inline void setToIndex(int64_t n) { _impl->setToIndex(); }

	inline T* data() const { return _impl->data(); }
	inline uint64_t index() const { 
#ifdef _DEBUG
		indexBoundsCheck();
#endif
		return _impl->index();
	}

	bool operator==(const Point& other) const { return _impl->operator==(*other._impl); }
	bool operator!=(const Point& other) const { return _impl->operator!=(*other._impl); }
	bool operator<(const Point& other) const { return _impl->operator<(*other._impl); }
	bool operator<=(const Point& other) const { return _impl->operator<=(*other._impl); }
	bool operator>(const Point& other) const { return _impl->operator>(*other._impl); }
	bool operator>=(const Point& other) const { return _impl->operator>=(*other._impl); }
private:
	std::unique_ptr<PointImpl<T>> _impl;

#ifdef _DEBUG
	inline void sqBracketBoundsCheck(uint64_t i) const {
		if (_impl->ptr < _impl->path->_data.data() || _impl->ptr >= _impl->path->_data.data() + _impl->path->_dataSize)
			throw std::out_of_range("Point is out of bounds for given path in Point::operator[]");

		if (i < 0 || i >= _impl->path->_dimension)
			throw std::out_of_range("Argument out of bounds in Point::operator[]");
	}

	inline void indexBoundsCheck() const {
		if (_impl->ptr < _impl->path->_data.data() || _impl->ptr >= _impl->path->_data.data() + _impl->path->_dataSize)
			throw std::out_of_range("Point is out of bounds for given path in Point::index()");
	}
#endif
};

#pragma pack(pop)
